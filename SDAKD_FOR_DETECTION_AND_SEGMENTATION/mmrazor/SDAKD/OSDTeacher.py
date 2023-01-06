from mmrazor.models.distillers.single_teacher import SingleTeacherDistiller
from mmrazor.models.builder import DISTILLERS,MODELS
import torch

@DISTILLERS.register_module()
class OSDTeacherDistiller(SingleTeacherDistiller):
    def __init__(self,
                 teacher,
                 teacher_trainable=False,
                 teacher_norm_eval=True,
                 components=tuple(),
                 **kwargs):
        super(OSDTeacherDistiller, self).__init__(
            teacher,
            teacher_trainable,
            teacher_norm_eval,
            components,
            **kwargs
        )
        self.convertor_training = False

    def set_convertor_training(self):
        self.convertor_training = True

    def unset_convertor_training(self):
        self.convertor_training = False

    def teacher_forward_pre_hook(self, module, input, same_indices):
        input = list(input)
        s_module_name = self.teacher_student_name_map[self.teacher_module2name[module]]
        for idx, item in zip(same_indices, self.student_inputs[s_module_name].pop(0)):
            if input[idx].shape != item.shape:
                input[idx].resize_(item.shape)
            input[idx].copy_(item)
        return tuple(input)

    def student_forward_pre_hook(self, module, input, same_indices):
        if not module.training and not self.convertor_training:
            return input
        same_input = []
        for idx in same_indices:
            same_input.append(input[idx])
        self.student_inputs[self.student_module2name[module]].append(same_input)
        return input

    def teacher_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training or self.convertor_training:
            self.teacher_outputs[self.teacher_module2name[module]].append(
                outputs)

    def student_forward_output_hook(self, module, inputs, outputs):
        """Save the module's forward output.

        Args:
            module (:obj:`torch.nn.Module`): The module to register hook.
            inputs (tuple): The input of the module.
            outputs (tuple): The output of the module.
        """
        if self.training or self.convertor_training:
            self.student_outputs[self.student_module2name[module]].append(
                outputs)


    def exec_teacher_forward(self, data):
        """Execute the teacher's forward function.

        After this function, the teacher's featuremaps will be saved in
        ``teacher_outputs``.
        """

        # Convert the context manager's mode to teacher.
        self.reset_ctx_teacher_mode(True)
        # Clear the saved data of the last forward。
        self.reset_outputs(self.teacher_outputs)

        if self.teacher_trainable or self.convertor_training:
            output = self.teacher(**data)
        else:
            with torch.no_grad():
                output = self.teacher(**data)

        return output
