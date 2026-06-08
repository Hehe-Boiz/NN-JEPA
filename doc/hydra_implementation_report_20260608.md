# Báo cáo triển khai Hydra cho NN-JEPA

Ngày cập nhật: 2026-06-08

## 1. Mục tiêu

Mục tiêu của thay đổi này là thêm cách train bằng file YAML/Hydra để không phải nhớ lệnh CLI rất dài.

Yêu cầu chính:

- giữ nguyên CLI cũ để không phá pipeline đang chạy
- thêm cấu hình train bằng YAML cho `tiny`, `small`, `base`
- có thể override tham số nhanh trên command line
- có dry-run để kiểm tra config trước khi train thật
- kiểm tra lại toàn bộ sau khi sửa

## 2. Các file đã thêm

```text
src/tools/train_rc_jepa_ac_features_hydra.py
configs/hydra/config.yaml
configs/hydra/experiment/rc_jepa_tiny.yaml
configs/hydra/experiment/rc_jepa_small.yaml
configs/hydra/experiment/rc_jepa_base.yaml
tests/test_hydra_train_config.py
doc/hydra_implementation_report_20260608.md
```

## 3. Các file đã sửa

```text
pyproject.toml
src/tools/train_rc_jepa_ac_features.py
```

Thay đổi trong `pyproject.toml`:

```text
thêm dependency: hydra-core>=1.3
```

Thay đổi trong `train_rc_jepa_ac_features.py`:

```text
main() giờ có thể nhận argparse.Namespace trực tiếp
CLI cũ vẫn chạy y như trước
Hydra wrapper gọi lại đúng train loop cũ, không clone logic train
```

Điểm này quan trọng: Hydra chỉ là lớp cấu hình bên ngoài. Model, Dataset, loss, scheduler, W&B, checkpoint, resume vẫn dùng đúng code train hiện tại.

## 4. Cách hoạt động của wrapper Hydra

File chính:

```text
src/tools/train_rc_jepa_ac_features_hydra.py
```

Luồng chạy:

```text
Hydra đọc YAML
-> wrapper chuyển YAML thành argparse.Namespace
-> gọi tools.train_rc_jepa_ac_features.main(args)
-> train loop cũ chạy như bình thường
```

Wrapper không tự train riêng, không tự viết loss riêng, không tự tạo model riêng.

## 5. Cấu trúc config Hydra

Config gốc:

```text
configs/hydra/config.yaml
```

Trong file này có:

```yaml
defaults:
  - experiment: rc_jepa_tiny
  - _self_

hydra:
  job:
    chdir: false
```

Điểm quan trọng:

```text
hydra.job.chdir: false
```

Hydra mặc định có thể đổi working directory sang thư mục run mới. Với project này, nếu đổi working directory thì các path tương đối như `data/processed/...` và `checkpoints/...` rất dễ bị lệch. Vì vậy đã tắt đổi directory.

Mỗi experiment hiện có thêm:

```yaml
runtime:
  dry_run: false
  require_cuda: true

train:
  device: auto
```

Ý nghĩa:

```text
device=auto:
  dry-run và train sẽ tự resolve sang cuda nếu CUDA sẵn sàng, ngược lại là cpu

require_cuda=true:
  khi train thật, nếu CUDA không sẵn sàng thì dừng bằng lỗi rõ ràng
  tránh việc model vô tình train trên CPU rất chậm
```

Nếu muốn debug CPU có chủ đích:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  runtime.require_cuda=false \
  train.device=cpu \
  wandb.mode=disabled
```

## 6. Các experiment YAML hiện có

### 6.1. Tiny

File:

```text
configs/hydra/experiment/rc_jepa_tiny.yaml
```

Thông số chính:

```text
model.size = tiny
predictor_dim = 128
predictor_depth = 2
predictor_heads = 4
params = 670,464
epochs = 20
warmup_epochs = 2
early_stopping_patience = 5
output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_tiny
```

### 6.2. Small

File:

```text
configs/hydra/experiment/rc_jepa_small.yaml
```

Thông số chính:

```text
model.size = small
predictor_dim = 256
predictor_depth = 4
predictor_heads = 4
params = 3,706,112
epochs = 50
warmup_epochs = 3
early_stopping_patience = 10
output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_small
```

### 6.3. Base

File:

```text
configs/hydra/experiment/rc_jepa_base.yaml
```

Thông số chính:

```text
model.size = base
predictor_dim = 512
predictor_depth = 6
predictor_heads = 8
params = 20,007,680
epochs = 100
warmup_epochs = 4
early_stopping_patience = 15
output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607
```

## 7. Lệnh chạy bằng Hydra

Trước khi chạy:

```bash
conda activate nn-jepa
```

### 7.1. Train tiny

`rc_jepa_tiny` là default, nên có thể chạy ngắn:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra
```

Hoặc ghi rõ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny
```

### 7.2. Train small

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_small
```

### 7.3. Train base

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base
```

## 8. Dry-run kiểm tra config trước khi train

Dry-run không train. Nó chỉ in ra `argparse.Namespace` cuối cùng sau khi Hydra YAML đã được map sang train args.

Tiny dry-run:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  runtime.dry_run=true \
  wandb.mode=disabled \
  train.device=cpu
```

Base dry-run:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base \
  runtime.dry_run=true \
  wandb.mode=disabled \
  train.device=cpu
```

Lưu ý: `train.device=cpu` trong dry-run chỉ để ép kiểm tra trên CPU. Config mặc định hiện dùng `train.device=auto` và `runtime.require_cuda=true`, nên train thật vẫn yêu cầu CUDA nếu không override có chủ đích.

## 9. Override tham số bằng Hydra

Ví dụ train tiny chỉ 5 epoch:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  train.epochs=5
```

Ví dụ tắt W&B:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  wandb.mode=disabled
```

Ví dụ đổi W&B entity/team:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  wandb.entity=https-ou-edu-vn
```

Ví dụ resume:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base \
  train.resume_from=checkpoints/rc_jepa_ac_vitb_features_20260607/last.pt
```

Nếu bị dừng giữa `val` và cần resume từ `last_train.pt`:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base \
  train.resume_from=checkpoints/rc_jepa_ac_vitb_features_20260607/last_train.pt
```

## 10. Cảnh báo khi override model size thủ công

Nếu dùng experiment tiny nhưng override:

```bash
model.size=small
```

thì `output_dir` vẫn là folder tiny nếu không override thêm.

Vì vậy nếu đổi model size thủ công, nên đổi luôn `output_dir` và tag W&B:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  model.size=small \
  output_dir=checkpoints/rc_jepa_ac_vitb_features_20260607_small_manual \
  wandb.tags='[small,hydra,manual]'
```

Khuyến nghị thực tế: dùng thẳng `experiment=rc_jepa_small` thay vì override `model.size`.

## 11. Kiểm tra config bằng `--cfg job`

Hydra cho phép in config cuối cùng mà không chạy train:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base \
  --cfg job
```

Lệnh này hữu ích để kiểm tra YAML trước khi train thật.

## 12. Dependency đã cài

Đã thêm và cài:

```text
hydra-core 1.3.2
omegaconf 2.3.0
```

Lệnh đã chạy:

```bash
conda run -n nn-jepa pip install -e .
```

Lần đầu chạy trong sandbox bị lỗi DNS vì không có network. Sau đó đã chạy lại với quyền mạng và cài thành công.

## 13. Kết quả kiểm tra

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
Ran 33 tests
OK
```

Đã chạy:

```bash
git diff --check
```

Kết quả:

```text
pass, không có lỗi whitespace
```

Đã chạy kiểm tra Hydra compose:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra --cfg job
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_base --cfg job
```

Kết quả:

```text
Hydra in ra đúng config tiny và base
```

Đã chạy kiểm tra dry-run:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  runtime.dry_run=true \
  wandb.mode=disabled \
  train.device=cpu
```

Kết quả tiny:

```text
model_size = tiny
predictor_dim = 128
predictor_depth = 2
predictor_heads = 4
output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_tiny
```

Kết quả base:

```text
model_size = base
predictor_dim = 512
predictor_depth = 6
predictor_heads = 8
output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607
```

Audit cuối cũng kiểm tra GPU:

```text
nvidia-smi: fail, không giao tiếp được với NVIDIA driver
torch.cuda.is_available(): False
torch.cuda.device_count(): 0
```

Vì vậy, ở trạng thái máy hiện tại, train thật bằng Hydra sẽ dừng với lỗi rõ ràng:

```text
Hydra config requires CUDA, but CUDA is not available.
```

Đây là hành vi mong muốn để tránh train nhầm CPU.

## 14. Kết luận

Hydra đã được thêm vào NN-JEPA theo hướng an toàn:

```text
CLI cũ vẫn giữ nguyên
train loop cũ vẫn được dùng lại
YAML config đã có tiny/small/base
dry-run đã có để kiểm tra trước khi train
test toàn bộ pass
dependency hydra-core đã cài trong env nn-jepa
```

Lệnh nên dùng để thử trước:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  runtime.dry_run=true \
  wandb.mode=disabled \
  train.device=cpu
```

Nếu dry-run đúng, train tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny
```

Khi tiny ổn, train base:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_base
```
