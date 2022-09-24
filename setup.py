from setuptools import find_packages, setup

setup(
    name="llacd",
    version="0.0.1",
    description="Learning Limit Augmentation in Knowledge Distillation",
    packages=find_packages(),
    install_requires=["timm", "Pillow", "tqdm", "wandb", "einops", "omegaconf"],
)
