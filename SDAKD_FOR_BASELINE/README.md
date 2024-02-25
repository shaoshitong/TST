## LLACD
Teach What You Should Teach: A Data-Based Distillation Method

## install

```bash
conda env create -f environment.yml
source activate llacd
```
## run

please modify the `data_path` and `local_ckpt_path` (you can download the ckpt from [checkpoint](https://github.com/shaoshitong/torchdistill/releases/tag/v0.3.3/))in config file.
```bash
python train_for_sdakd.py --config_file configs/{name} --cuda_devices 0
```