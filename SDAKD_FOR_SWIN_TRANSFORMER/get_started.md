```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main_for_sdakd.py \
--cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path <imagenet-path> --batch-size 128
```