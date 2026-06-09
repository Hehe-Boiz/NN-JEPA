# Hướng dẫn train feature cache bằng Hydra

Ngày viết: 2026-06-09

Mục tiêu của file này:

- Chuyển lệnh train dài `tools.train_rc_jepa_ac_features` sang Hydra.
- Phân biệt rõ train sạch trên data mới và resume/fine-tune từ checkpoint cũ.
- Ghi rõ tham số nào ảnh hưởng kết quả, tham số nào chủ yếu ảnh hưởng tốc độ.

## 1. Batch/worker có ảnh hưởng kết quả không?

### Khi extract feature

Trong lệnh:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features ...
```

`--batch-size` là số frame đưa qua encoder V-JEPA trong một lần forward.

`--num-workers` là số worker đọc ảnh/chuẩn bị batch.

Với feature extraction:

- `batch-size` không đổi feature nếu cùng checkpoint, cùng encoder, cùng dtype, cùng preprocessing.
- `num-workers` không đổi feature.
- Hai tham số này chủ yếu ảnh hưởng tốc độ, RAM/CPU và VRAM.
- Nếu OOM thì giảm `batch-size`.
- Nếu GPU rảnh vì chờ data thì tăng `num-workers`.

Khuyến nghị ban đầu cho ViT-B 384:

```bash
--batch-size 32
--num-workers 8
```

Nếu OOM:

```bash
--batch-size 16
```

### Khi train predictor

Trong lệnh:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features ...
```

`--batch-size` là batch train predictor. Tham số này **có thể ảnh hưởng kết quả** vì nó đổi gradient update, noise của optimizer, và số step mỗi epoch.

`--num-workers` chỉ là số worker DataLoader. Tham số này **không có ý nghĩa học thuật trực tiếp**, chủ yếu ảnh hưởng tốc độ nạp data. Trong thực tế GPU training vẫn có thể không bitwise-identical tuyệt đối do CUDA/kernel, nhưng về mặt model quality thì `num-workers` không phải hyperparameter học chính.

Vì vậy:

- Muốn so sánh công bằng thì giữ nguyên `batch_size`, `lr`, `seed`, split, feature cache.
- Có thể đổi `num_workers` để tăng tốc nếu cần.
- Không nên đổi `batch_size` giữa các run rồi so loss như cùng một cấu hình.

## 2. Không nên resume checkpoint cũ nếu data đã đổi và muốn run sạch

Lệnh cũ của bạn có:

```bash
--resume-from checkpoints/rc_jepa_ac_vitb_features/last.pt
--wandb-run-id 8an0pxpo
--wandb-resume allow
```

Ý nghĩa:

- `--resume-from`: load lại predictor, optimizer, scheduler, history, best loss, global step.
- `--wandb-run-id`: log tiếp vào đúng W&B run cũ.
- `--wandb-resume allow`: cho W&B nối tiếp run cũ.

Nếu data đã đổi, dùng các tham số này nghĩa là:

```text
fine-tune tiếp từ checkpoint/log cũ trên data mới
```

Không phải:

```text
train sạch từ đầu trên data mới
```

Với data mới, khuyến nghị mặc định là train sạch:

- Không truyền `train.resume_from`.
- Không truyền `wandb.run_id`.
- Dùng `output_dir` mới.
- Đặt `wandb.run_name` mới.
- Thêm tag `new-data`.

Chỉ dùng resume cũ nếu bạn cố tình muốn tiếp tục training một model cũ.

## 3. Lệnh Hydra train sạch cho data mới

Nếu muốn chạy bản `tiny` để thử nhanh trước, dùng:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata
```

Config này nằm ở:

```text
configs/hydra/experiment/rc_jepa_tiny_newdata.yaml
```

Nó dùng:

```text
model.size = tiny
predictor_dim = 128
predictor_depth = 2
predictor_heads = 4
output_dir = checkpoints/rc_jepa_ac_vitb_features_newdata_tiny
epochs = 20
batch_size = 10
eval_batch_size = 2
```

Nếu muốn chạy bản `base` cho data mới sau khi tiny ổn, dùng:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base_newdata
```

Config base nằm ở:

```text
configs/hydra/experiment/rc_jepa_base_newdata.yaml
```

Nó dùng:

```text
model.size = base
predictor_dim = 512
predictor_depth = 6
predictor_heads = 8
output_dir = checkpoints/rc_jepa_ac_vitb_features_newdata
epochs = 100
batch_size = 10
eval_batch_size = 2
```

Nếu muốn W&B entity/team cụ thể thì chỉ override đúng field đó:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  wandb.entity=<team_or_entity>
```

Ví dụ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  wandb.entity=https-ou-edu-vn
```

## 4. Dry-run trước khi train thật

Nên chạy dry-run để kiểm tra Hydra đã map đúng args:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  runtime.dry_run=true \
  runtime.require_cuda=false \
  wandb.disabled=true \
  train.device=cpu
```

Dry-run chỉ in config, không train.

## 5. Lệnh Hydra nếu cố tình resume/fine-tune run cũ

Chỉ dùng lệnh này nếu bạn muốn nối tiếp đúng run W&B cũ `8an0pxpo` và checkpoint cũ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base \
  output_dir=checkpoints/rc_jepa_ac_vitb_features \
  data.features_dir=data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  data.manifest_dir=data/processed/manifests \
  train.resume_from=checkpoints/rc_jepa_ac_vitb_features/last.pt \
  train.epochs=100 \
  train.batch_size=10 \
  train.eval_batch_size=2 \
  train.num_workers=8 \
  train.lr=1.0e-4 \
  train.warmup_epochs=4 \
  train.warmup_start_factor=0.1 \
  train.min_lr_ratio=0.1 \
  train.early_stopping_patience=15 \
  wandb.project=nn-jepa-rc \
  wandb.run_id=8an0pxpo \
  wandb.continue_run=true \
  wandb.resume=allow \
  wandb.log_every=20 \
  wandb.watch_log=all \
  wandb.watch_freq=100 \
  wandb.grad_stats_every=10 \
  wandb.param_stats_every=100
```

Nếu muốn resume/fine-tune từ checkpoint nhưng tạo W&B run mới, không nối log vào run cũ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.resume_from=checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/last.pt \
  wandb.continue_run=false
```

Quy tắc:

```text
train.resume_from != null      -> load checkpoint model/optimizer/scheduler để train tiếp
wandb.continue_run=true        -> nếu có run id thì nối log vào W&B run cũ
wandb.continue_run=false       -> vẫn load checkpoint, nhưng tạo W&B run mới
wandb.run_id=<id>              -> chỉ dùng khi muốn ép nối vào đúng run cụ thể
```

Cảnh báo:

- Nếu data mới khác data cũ, curve W&B sẽ nối tiếp nhưng không còn là một thí nghiệm sạch.
- Scheduler/optimizer cũng tiếp tục từ checkpoint cũ.
- `best_val_loss` và early stopping history cũng là history cũ.
- Dùng cách này cho fine-tune, không dùng cho benchmark data mới.

## 6. Lệnh train `official_lite tiny` bằng Hydra

Nếu muốn thử predictor gần source V-JEPA AC hơn:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny \
  output_dir=checkpoints/rc_jepa_ac_vitb_features_newdata_official_lite_tiny \
  data.features_dir=data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  data.manifest_dir=data/processed/manifests \
  train.batch_size=4 \
  train.eval_batch_size=1 \
  train.num_workers=8 \
  wandb.run_name=rc-jepa-official-lite-tiny-newdata \
  'wandb.tags=[official_lite,tiny,hydra,new-data]'
```

`official_lite` nên chạy batch nhỏ hơn vì sequence vẫn là full token:

```text
8 frame * (576 patch token + 2 cond token) = 4624 token/sample
```

## 7. Checklist trước khi train

Trước khi train thật:

1. `nvidia-smi` phải hoạt động.
2. Feature cache phải đủ session theo manifest.
3. Chạy dry-run Hydra.
4. Dùng output dir mới nếu data mới.
5. Không dùng `wandb.run_id` cũ nếu muốn run sạch.

Kiểm tra cache còn thiếu session hay không:

```bash
PYTHONPATH=src python3 -c "from pathlib import Path; import json; m=Path('data/processed/manifests'); f=Path('data/processed/features/vjepa2_1_vitb_384_ema_fp32/sessions'); s={json.loads(line)['session_id'] for split in ('train','val','test') for line in (m/f'{split}.jsonl').read_text().splitlines() if line.strip()}; j={p.stem for p in f.glob('*.json')}; n={p.stem for p in f.glob('*.npy')}; print({'manifest_sessions':len(s),'json':len(j),'npy':len(n),'missing_json':len(s-j),'missing_npy':len(s-n)}); print(sorted(s-j)[:20])"
```

Nếu `missing_json` hoặc `missing_npy` lớn hơn `0`, cần extract bù feature trước.
