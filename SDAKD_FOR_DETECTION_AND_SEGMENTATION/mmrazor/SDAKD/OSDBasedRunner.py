from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
import os.path as osp
import platform
import shutil
import time
import mmcv
import yaml
from mmcv.runner import HOOKS, RUNNERS, EpochBasedRunner
from typing import Any, Dict, List, Optional, Tuple
import torch
import warnings
from mmcv.runner.utils import get_host_info
from mmcv.utils import TORCH_VERSION, digit_version

class ALRS:
    """
    proposer: Huanran Chen
    theory: landscape
    Bootstrap Generalization Ability from Loss Landscape Perspective
    """

    def __init__(self, optimizer, loss_threshold=0.02, loss_ratio_threshold=0.02, decay_rate=0.9):
        self.optimizer = optimizer
        self.loss_threshold = loss_threshold
        self.decay_rate = decay_rate
        self.loss_ratio_threshold = loss_ratio_threshold

        self.last_loss = 999

    def step(self, loss):
        delta = self.last_loss - loss
        if (
            delta < self.loss_threshold
            and abs(delta / (self.last_loss - 1e-6)) < self.loss_ratio_threshold
        ):
            for group in self.optimizer.param_groups:
                group["lr"] *= self.decay_rate
                now_lr = group["lr"]
                print(f"now lr = {now_lr}")

        self.last_loss = loss


@RUNNERS.register_module()
class OSDBasedRunner(EpochBasedRunner):
    def __init__(self,
                 model,
                 batch_processor=None,
                 optimizer=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        super(OSDBasedRunner,self).__init__(
            model,
            batch_processor,
            optimizer,
            work_dir,
            logger,
            meta,
            max_iters,
            max_epochs
        )
        self.c_optimizer = torch.optim.SGD(self.model.module.convertor.parameters(), lr=0.01, momentum=0.9)
        self.c_scheduler = ALRS(self.c_optimizer)
        self.c_scaler = torch.cuda.amp.GradScaler()

    def reset_convertor(self):
        del self.c_optimizer
        del self.c_scaler
        del self.c_scheduler
        self.c_optimizer = torch.optim.SGD(self.model.module.convertor.parameters(), lr=0.01, momentum=0.9)
        self.c_scheduler = ALRS(self.c_optimizer)
        self.c_scaler = torch.cuda.amp.GradScaler()

    def run_convertor_iter(self, data_batch, **kwargs):
        if "FCOS" not in self.model.module.architecture.model.__class__.__name__:
            self.model.module.architecture.eval()
        self.model.module.architecture.requires_grad_(False)
        self.model.module.distiller.teacher.requires_grad_(False)
        inputs, kwargs = self.model.scatter([data_batch,self.optimizer], kwargs, self.model.device_ids)
        if len(self.model.device_ids) == 1:
            output = self.model.module.train_convertor_step(*inputs[0], **kwargs[0])
        else:
            outputs = self.model.parallel_apply(
                self.model._module_copies[:len(inputs)], inputs, kwargs)
            output = self.model.gather(outputs, self.output_device)
        if torch.is_grad_enabled() and getattr(
                self.model, 'require_backward_grad_sync', True):
            if self.model.find_unused_parameters:
                from torch.nn.parallel.distributed import _find_tensors
                self.model.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.model.reducer.prepare_for_backward([])
        else:
            if ('parrots' not in TORCH_VERSION
                    and digit_version(TORCH_VERSION) > digit_version('1.2')):
                self.require_forward_param_sync = False
        if "FCOS" not in self.model.module.architecture.model.__class__.__name__:
            self.model.module.architecture.train()
        self.model.module.architecture.requires_grad_(True)
        self.model.module.distiller.teacher.requires_grad_(True)
        self.c_outputs = output

    def train_convertor(self, data_loader, **kwargs):
        self.model.train()
        self.mode = 'train'
        self.data_loader = data_loader
        self._max_iters = self._max_epochs * len(self.data_loader)
        self.call_hook('before_train_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        self.c_total_loss = 0
        self.c_total_iter = 0
        for i, data_batch in enumerate(self.data_loader):
            self.call_hook('before_train_iter')
            self.run_convertor_iter(data_batch, **kwargs)
            self.c_optimizer.zero_grad()
            self.c_scaler.scale(self.c_outputs['loss']).backward()
            self.c_scaler.unscale_(self.c_optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.module.convertor.parameters(),20)
            self.c_scaler.step(self.c_optimizer)
            self.c_scaler.update()
            self.c_total_loss+=self.c_outputs['loss'].item()
            self.c_total_iter+=1
            if self.c_total_iter%100==0:
                self.logger.info('convertor loss: %s, iter: %d', round(self.c_total_loss/self.c_total_iter,4), self.c_total_iter)
                magnitude_str = self.model.module.convertor.module.print_magnitudes()
                self.logger.info(magnitude_str)
                probability_str = self.model.module.convertor.module.print_probabilities()
                self.logger.info(probability_str)
                self.logger.info(self.c_outputs["log_vars"])

        c_average_loss = round(self.c_total_loss/self.c_total_iter,4)
        self.c_scheduler.step(c_average_loss)


    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, epochs) to specify the
                running order and epochs. E.g, [('train', 2), ('val', 1)] means
                running 2 epochs for training and 1 epoch for validation,
                iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_epochs is not None:
            warnings.warn(
                'setting max_epochs in run is deprecated, '
                'please set max_epochs in runner_config', DeprecationWarning)
            self._max_epochs = max_epochs

        assert self._max_epochs is not None, (
            'max_epochs must be specified during instantiation')

        for i, flow in enumerate(workflow):
            mode, epochs = flow
            if mode == 'train':
                self._max_iters = self._max_epochs * len(data_loaders[i])
                break

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d epochs', workflow,
                         self._max_epochs)
        self.call_hook('before_run')

        while self.epoch < self._max_epochs:
            if self.epoch in self.model.module.convertor_training_epoch:
                p=-1
                for i,flow in enumerate(workflow):
                    mode, epochs = flow
                    if mode == "train" and self.epoch < self._max_epochs:
                        p = i
                    else:
                        continue
                if p==-1:
                    raise NotImplementedError("no training workflow!")
                self.model.module.set_convertor(self.model.module)
                self.reset_convertor()
                self.logger.info("begin training convertor!")
                for _ in range(self.model.module.convertor_epoch_number):
                    self.logger.info("convertor epoch %s begin",_)
                    self.train_convertor(data_loaders[p],**kwargs)
                    self.logger.info("convertor epoch %s end",_)

                self.model.module.unset_convertor(self.model.module)

            for i, flow in enumerate(workflow):
                mode, epochs = flow
                if isinstance(mode, str):  # self.train()
                    if not hasattr(self, mode):
                        raise ValueError(
                            f'runner has no method named "{mode}" to run an '
                            'epoch')
                    epoch_runner = getattr(self, mode)
                else:
                    raise TypeError(
                        'mode in workflow must be a str, but got {}'.format(
                            type(mode)))

                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')