# Point-Adaptive Diffusion Distillation for 3D Point Cloud Semantic Segmentation
## Dataset

```text
PADD/
├── configs/
│   ├── nuscenes/          # 16 classes, 4 input channels
│   └── s3dis/             # 13 classes, 6 input channels
├── pointcept/
│   ├── models/point_transformer_v3/
│   ├── models/diffusion_kd/
│   ├── engines/           # training, testing, and distillation
│   └── datasets/          # nuScenes, S3DIS, and preprocessing
├── libs/pointops/         # CUDA KNN used during evaluation
├── scripts/train.sh
├── scripts/test.sh
└── tools/
```

## Environment

Linux, CUDA 12.4, and a matching PyTorch 2.5 build are required. From this directory:

```bash
conda env create -f environment.yml
conda activate pointcept-torch2.5.0-cu12.4
```

`environment.yml` installs `spconv`, `torch-scatter`, FlashAttention, and `libs/pointops`. The provided configs set `enable_flash` to `True`. If FlashAttention cannot be installed, set `enable_flash` to `False` in the corresponding config.

The diffusion module does not depend on the CUDA extensions. Check it with:

```bash
python -m unittest tests/test_diffusion_kd.py
```

## Data

Config paths are relative to this directory: `data/nuscenes` and `data/s3dis`. Download the raw datasets under their own licenses.

nuScenes preprocessing depends on `nuscenes-devkit`. It writes `info/nuscenes_infos_10sweeps_{train,val,test}.pkl` under `output_root`. `data_root` in the config must be able to read that `info` directory and the point clouds recorded in the pickles.

```bash
python pointcept/datasets/preprocessing/nuscenes/preprocess_nuscenes_info.py \
  --dataset_root /path/to/nuscenes \
  --output_root data/nuscenes \
  --max_sweeps 10
```

S3DIS uses Stanford3D Area_1 through Area_6. Training uses Area_1/2/3/4/6, and validation uses Area_5.

```bash
python pointcept/datasets/preprocessing/s3dis/preprocess_s3dis.py \
  --dataset_root /path/to/Stanford3dDataset_v1.2 \
  --output_root data/s3dis \
  --splits Area_1 Area_2 Area_3 Area_4 Area_5 Area_6 \
  --parse_normal
```

## Training

Run the commands from this directory. `-g` is the number of GPUs, and `-v` is `CUDA_VISIBLE_DEVICES`. Logs and weights are written to `exp/<dataset>/<experiment>/`.

Train the teacher first:

```bash
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base
```

Place the resulting `model/model_best.pth` at the `teacher_model_ckpt` path, or edit the config:

- nuScenes: `checkpoints/nuscenes_ptv3/model_best.pth`
- S3DIS: `checkpoints/s3dis_ptv3/model_best.pth`

Train a standalone student without distillation:

```bash
sh scripts/train.sh -g 2 -d nuscenes -c semseg-pt-v3m1-0-base_student -n semseg-pt-v3m1-0-base_student
sh scripts/train.sh -g 2 -d s3dis -c semseg-pt-v3m1-0-student -n semseg-pt-v3m1-0-student
```

`PADD.enabled` is already on in the distillation configs. Logit and feature alignment are enabled from the command line. This nuScenes example uses logits, features, and PADD together:

```bash
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v3m1-0-base_distill \
  -n semseg-pt-v3m1-0-base_distill \
  -a logits -e 1. -t 4. -b feat -f 1.
```

S3DIS:

```bash
sh scripts/train.sh -g 4 -d s3dis -c semseg-pt-v3m1-0-distill \
  -n semseg-pt-v3m1-0-distill \
  -a logits -e 1. -t 2.
```

Common arguments:

| Argument | Meaning |
| --- | --- |
| `-a logits` | Enable logit KL distillation |
| `-b feat` | Enable feature MSE distillation |
| `-e`, `-f` | Weights of the two distillation losses |
| `-t` | Logit temperature |
| `-W`, `-A` | PADD noise-loss weight and feature-alignment weight; both default to 1 |
| `-M joint` | Diffusion training mode: `joint`, `diffusion_pretrain`, or `student_refine` |
| `-k` | Align denoised student features with KL |
| `-l` | Enable memory-bank point-to-point relational distillation |
| `-V` | Save point clouds of teacher, student, and denoised features |
| `-r true` | Resume from an existing experiment directory |

Resume example:

```bash
sh scripts/train.sh -g 4 -d nuscenes -c semseg-pt-v3m1-0-base_distill \
  -n semseg-pt-v3m1-0-base_distill -r true \
  -a logits -e 1. -t 4. -b feat -f 1.
```

## Testing

The default weight name is `model_best`, which resolves to `exp/<dataset>/<experiment>/model/model_best.pth`.

```bash
sh scripts/test.sh -g 1 -d nuscenes -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base -w model_best
sh scripts/test.sh -g 1 -d s3dis -c semseg-pt-v3m1-0-base -n semseg-pt-v3m1-0-base -w model_best
```

## Citation

Please cite the PTv3 backbone:

```bibtex
@inproceedings{wu2024ptv3,
  title={Point Transformer V3: Simpler, Faster, Stronger},
  author={Wu, Xiaoyang and Jiang, Li and Wang, Peng-Shuai and Liu, Zhijian and Liu, Xihui and Qiao, Yu and Ouyang, Wanli and He, Tong and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```

This code is based on Pointcept and is released under the MIT license. See `LICENSE`.
