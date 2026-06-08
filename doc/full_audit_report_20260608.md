# Báo cáo audit toàn bộ NN-JEPA trước giai đoạn train quan trọng

Ngày audit: 2026-06-08

Mục tiêu: kiểm tra lại code, Hydra config, data raw, manifest, feature cache, DataLoader, predictor params, web tools và trạng thái repo trước khi chạy train thật.

## 1. Kết luận ngắn

Các phần đã kiểm tra và pass:

```text
code compile: pass
unit tests: 38/38 pass
git diff whitespace: pass
Hydra config compose: pass
Hydra dry-run tiny/small/base: pass
web job progress: pass
V-JEPA feature presets: pass
feature cache: complete, không cần extract lại
raw data: 63 session mới, không có session cũ 20260605 trong data/raw
manifest: 88,120 sample, không thiếu frame_path
feature cache vs manifest: khớp 88,120 frame, thiếu feature = 0
DataLoader windows: train 49,422, val 8,983, test 10,678
```

Blocker hiện tại:

```text
GPU/CUDA hiện không khả dụng trên máy ở thời điểm audit.
nvidia-smi không giao tiếp được với NVIDIA driver.
torch.cuda.is_available() = False.
```

Vì vậy, train thật bằng Hydra sẽ dừng rõ ràng nếu chưa sửa GPU/driver. Đây là hành vi đúng, vì không nên train nhầm trên CPU.

## 2. Kiểm tra môi trường Python

Lệnh kiểm tra:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -c "import torch, hydra, omegaconf, wandb; ..."
```

Kết quả:

```text
torch = 2.12.0+cu130
hydra = 1.3.2
omegaconf = 2.3.0
wandb = 0.27.2
torch.version.cuda = 13.0
torch.cuda.is_available() = False
torch.cuda.device_count() = 0
```

Kiểm tra driver:

```bash
nvidia-smi
```

Kết quả:

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.
```

Ý nghĩa:

- package PyTorch có CUDA build
- nhưng driver/NVML hiện không hoạt động hoặc GPU không được expose
- trước khi train thật cần sửa điểm này

## 3. Kiểm tra code và test

Đã chạy:

```bash
conda run -n nn-jepa python -m compileall -q src tests
```

Kết quả:

```text
pass
```

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m unittest discover -s tests -v
```

Kết quả:

```text
Ran 38 tests
OK
```

Các nhóm test đã cover:

```text
pipeline CSV/state/action
sensor synced CSV
outlier filter
sequence window không vượt session/gap
feature sequence dataset
predictor shape
world model losses
rollout không dùng state tương lai thật
LR scheduler/warmup/resume
early stopping
W&B resume run id
web viewer job commands
web viewer progress parser
V-JEPA feature preset resolver
V-JEPA feature preset web command
Hydra config mapping
```

Kiểm tra whitespace diff:

```bash
git diff --check
```

Kết quả:

```text
pass, không có lỗi whitespace
```

## 4. Kiểm tra Hydra

Các config hiện có:

```text
configs/hydra/config.yaml
configs/hydra/experiment/rc_jepa_tiny.yaml
configs/hydra/experiment/rc_jepa_small.yaml
configs/hydra/experiment/rc_jepa_base.yaml
```

Hydra help nhận đúng group:

```text
experiment: rc_jepa_base, rc_jepa_small, rc_jepa_tiny
```

Đã kiểm tra:

```bash
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra --cfg job
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_base --cfg job
```

Kết quả:

```text
Hydra compose được config tiny/base đúng.
```

Đã dry-run:

```bash
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra runtime.dry_run=true wandb.mode=disabled
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_small runtime.dry_run=true wandb.mode=disabled
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_base runtime.dry_run=true wandb.mode=disabled
```

Kết quả mapping predictor:

```text
tiny:
  predictor_dim = 128
  predictor_depth = 2
  predictor_heads = 4
  output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_tiny

small:
  predictor_dim = 256
  predictor_depth = 4
  predictor_heads = 4
  output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_small

base:
  predictor_dim = 512
  predictor_depth = 6
  predictor_heads = 8
  output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607
```

Đã kiểm tra guard CUDA:

```bash
PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  wandb.mode=disabled \
  train.epochs=0
```

Kết quả:

```text
RuntimeError: Hydra config requires CUDA, but CUDA is not available.
```

Đây là kết quả đúng. Nếu GPU chưa sẵn sàng, train thật phải dừng thay vì chạy CPU.

## 5. Kiểm tra raw data và manifest

Kết quả audit:

```text
raw_sessions = 63
old_20260605_raw = 0
missing_synced = 0
```

Manifest:

```text
train samples = 62,501
val samples   = 11,379
test samples  = 14,240
total samples = 88,120
```

Số session theo split:

```text
train sessions = 45
val sessions   = 9
test sessions  = 9
```

Kiểm tra split:

```text
session overlap train/val/test = 0
missing frame_path = 0
missing state/action column = 0
```

State/action train JEPA-AC hiện dùng:

```text
state_columns:
  yaw_rate_t
  accel_x_t
  accel_y_t
  steering_last_t
  throttle_last_t

action_columns:
  steering_cmd_t
  throttle_cmd_t
```

## 6. Kiểm tra feature cache

Feature cache path:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

Metadata:

```text
session_count = 63
frame_count = 88,120
tokens_per_frame = 576
embed_dim = 768
dtype = fp32
encoder = vit_base_384
checkpoint_key = ema_encoder
```

File cache:

```text
.npy files = 63
.json files = 63
feature_frame_count_from_npy = 88,120
manifest_frames_missing_feature = 0
extra_feature_frames_not_in_manifest = 0
```

Đã chạy extractor ở chế độ kiểm tra cache:

```bash
PYTHONPATH=src python -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --batch-size 32 \
  --dtype fp32 \
  --num-workers 4
```

Kết quả:

```text
status = feature_cache_already_complete
skipped_compatible = 63
extracted = 0
missing = 0
incompatible = 0
```

Nghĩa là không cần extract lại trước khi train.

## 7. Kiểm tra DataLoader và predictor params

Đã build DataLoader từ feature cache với:

```text
batch_size = 10
eval_batch_size = 2
num_workers = 0 trong audit
```

Kết quả windows:

```text
train windows = 49,422
val windows   = 8,983
test windows  = 10,678
```

Feature shape:

```text
tokens_per_frame = 576
embed_dim = 768
```

Normalizer:

```text
state_normalizer = True
action_normalizer = True
```

Predictor params:

```text
tiny  = 670,464
small = 3,706,112
base  = 20,007,680
```

## 8. Kiểm tra web/session tools

Đã chạy:

```bash
PYTHONPATH=src python -m tools.session_web_viewer --help
```

Kết quả:

```text
CLI load được, option host/port đúng.
```

Unit test web cũng pass:

```text
build_sync_command
build_preprocess_command
build_extract_feature_command
build_job_command validation
session payload/frame sorting
job progress helper/parser
progress percent clamp
```

Smoke test progress:

```bash
PYTHONPATH=src python -m tools.sync_drive_data \
  --dry-run \
  --skip-rclone-copy \
  --skip-extract \
  --skip-sensor-sync
```

Kết quả: phát progress marker `0%` và `100%`.

```bash
PYTHONPATH=src python -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --batch-size 32 \
  --dtype fp32 \
  --num-workers 4
```

Kết quả: `feature_cache_already_complete`, `skipped_compatible=63`, progress `100%`.

## 9. Kiểm tra repo JEPA

Repo `JEPA/`:

```text
HEAD = be50793 Add visual navigation (TopoGraph subgoal) + control on current servo
```

Local dirty bên trong `JEPA/`:

```text
D data/.gitkeep
```

Đây là do thư mục data local đã được dùng/staging. Không ảnh hưởng NN-JEPA train, nhưng nếu muốn commit riêng repo `JEPA/` thì cần xử lý file này.

## 10. Trạng thái git NN-JEPA

Có các file đã sửa hoặc thêm nhưng chưa commit.

Tracked modified:

```text
HANDOFF.md
MEMORY.md
README.md
pyproject.toml
src/models/rc_jepa_ac.py
src/tools/train_rc_jepa_ac.py
src/tools/train_rc_jepa_ac_features.py
tests/test_rc_jepa_ac.py
```

Untracked đáng chú ý:

```text
configs/hydra/
src/tools/train_rc_jepa_ac_features_hydra.py
tests/test_hydra_train_config.py
doc/hydra_implementation_report_20260608.md
doc/jepa_update_20260608_hydra.md
doc/full_audit_report_20260608.md
JEPA/
vjepa2/
wandb/
```

Lưu ý:

- `JEPA/`, `vjepa2/`, `wandb/` là thư mục local lớn/untracked theo trạng thái hiện tại.
- Nếu chuẩn bị commit, cần chọn kỹ file nào add vào git, tránh add `wandb/` hoặc repo clone lồng nhau nếu không muốn.

## 11. Rủi ro còn lại

Không có audit nào đảm bảo tuyệt đối không còn bug, nhưng các lỗi hệ thống dễ thấy đã được kiểm tra. Rủi ro còn lại:

```text
1. GPU/driver đang lỗi hoặc không expose được GPU.
2. Chưa chạy train thật sau khi thêm Hydra vì CUDA hiện không khả dụng.
3. Chưa kiểm tra chất lượng model bằng metric sau train tiny/base mới.
4. Untracked files cần được quản lý cẩn thận trước khi commit.
```

Rủi ro quan trọng nhất hiện tại là GPU.

## 12. Việc cần làm trước khi train thật

1. Sửa/kiểm tra GPU:

```bash
nvidia-smi
```

phải chạy được.

2. Kiểm tra PyTorch thấy CUDA:

```bash
PYTHONPATH=src python3 - <<'PY'
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

3. Dry-run lại Hydra:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  runtime.dry_run=true \
  wandb.mode=disabled
```

Khi GPU ổn, dry-run phải in:

```text
device = cuda
```

4. Train tiny trước:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny
```

5. Khi tiny ổn, train base:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base
```

## 13. Kết luận cuối

Pipeline code, config, manifest, feature cache và DataLoader hiện đã qua audit.

Điểm duy nhất không đạt để train thật ngay lúc này là:

```text
GPU/CUDA không khả dụng do NVIDIA driver/NVML.
```

Do đã thêm `runtime.require_cuda=true`, Hydra sẽ không cho train thật nếu GPU chưa sẵn sàng. Điều này bảo vệ khỏi lỗi train nhầm CPU.
