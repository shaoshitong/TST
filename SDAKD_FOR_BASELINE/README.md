## LLACD
Using Learning Limit Augmentation to Select Appropriate Samples to Assist Distillation

## install

```bash
conda env create -f environment.yml
source activate llacd
```
## run

```bash
python train_for_sdakd.py --config_file configs/{name} --cuda_devices 0,1,2,3,4,5,6,7
```