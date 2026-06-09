# Báo cáo triển khai experiment dùng data servo cũ

Ngày viết: 2026-06-09

Phạm vi: repo `NN-JEPA`. Repo `JEPA/` chỉ được đọc để lấy data/staging, không sửa code gốc trong `JEPA/` hoặc `vjepa2/`.

## 1. Mục tiêu

Mục tiêu của experiment này là thử tận dụng lại data servo cũ `KDS 680HV` như một experiment riêng, không làm bẩn baseline data hiện tại.

Baseline hiện tại vẫn giữ nguyên:

```text
data/raw
data/processed
data/processed/features/vjepa2_1_vitb_384_ema_fp32
checkpoints/rc_jepa_ac_vitb_features_20260607
```

Experiment mới dùng root riêng:

```text
data/experiments/servo_old_mix_v1
data/experiments/servo_old_only_v1
```

Ý nghĩa:

- `servo_old_mix_v1`: trộn data servo hiện tại trong `data/raw` với data servo cũ.
- `servo_old_only_v1`: chỉ dùng data servo cũ để kiểm tra domain cũ riêng.
- Không copy data cũ vào `data/raw`.
- Không ghi đè `data/processed` baseline.
- Không dùng lại checkpoint train cũ để resume, vì dữ liệu và split đã đổi. Nên train fresh.

## 2. Nguồn data

Data hiện tại:

```text
data/raw/session_*
```

Data servo cũ:

```text
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV/session_*
```

Audit hiện tại:

```text
data/raw: 100 session hiện tại
data servo cũ KDS 680HV: 28 session cũ
```

Tất cả 28 session servo cũ preprocess được, không có session `status != ok`.

## 3. File code đã thêm/sửa

### 3.1. Tool build experiment dataset

File mới:

```text
src/tools/build_servo_experiment_dataset.py
```

Tool này làm các việc sau:

1. Chọn session theo mode:

```text
mixed       = data/raw + data servo cũ
old-only    = chỉ data servo cũ
current-only = chỉ data/raw nhưng ghi ra experiment root riêng
```

2. Chạy lại `preprocess_one_session` cho từng session nhưng tạm thời đổi output root sang experiment root.

3. Gắn metadata domain vào từng sample:

```json
{
  "data_domain": "current_servo",
  "source_raw_root": "...",
  "servo_experiment_mode": "mixed"
}
```

4. Chia train/val theo session và theo từng domain. Không tạo test split độc lập; `test.jsonl` được ghi bằng chính `val.jsonl` để tương thích tool cũ.

Điểm quan trọng: split theo domain giúp mixed experiment không bị trường hợp old servo chỉ nằm toàn bộ ở train hoặc toàn bộ ở val.

### 3.2. DataLoader trả thêm `data_domain`

Các file đã cập nhật:

```text
src/data/sequence_dataset.py
src/data/feature_sequence_dataset.py
```

Mỗi sample sequence giờ trả thêm:

```python
"data_domain": first_sample.get("data_domain", "unknown")
```

Nhờ vậy `val` có thể tách metric theo domain ngay trong train. `test` vẫn có thể tách domain nếu chạy eval/test riêng hoặc bật `--run-test`, nhưng hiện `test` chỉ là alias của `val`, không phải tập đánh giá độc lập.

### 3.3. Train loop log metric theo domain

File đã cập nhật:

```text
src/tools/train_rc_jepa_ac_features.py
```

Trong train loop feature-cache:

- Train vẫn log loss tổng như cũ.
- Val/test sẽ tính thêm loss theo từng `data_domain`; test hiện là val alias.

Metric sinh ra có dạng:

```text
val/domain/current_servo/loss
val/domain/current_servo/teacher_forcing_loss
val/domain/current_servo/rollout_loss
val/domain/old_servo/loss
val/domain/old_servo/teacher_forcing_loss
val/domain/old_servo/rollout_loss
test/domain/current_servo/loss  # chỉ khi chạy test riêng hoặc --run-test, hiện là val alias
test/domain/old_servo/loss      # chỉ khi chạy test riêng hoặc --run-test, hiện là val alias
```

Ý nghĩa: mixed experiment sẽ biết model đang tốt trên servo hiện tại nhưng tệ trên servo cũ, hoặc ngược lại.

### 3.4. Feature extractor có seed cache

File đã cập nhật:

```text
src/tools/extract_vjepa_features.py
```

Tham số mới:

```bash
--seed-from-features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

Tác dụng:

- Với mixed experiment, 100 session hiện tại đã có feature cache baseline.
- Tool sẽ symlink `.npy + .json` tương ứng vào feature dir experiment nếu metadata khớp.
- Chỉ encode thêm session cũ chưa có cache.

Điều kiện để seed cache được chấp nhận:

```text
format_version
feature_layout
encoder_name
checkpoint_key
image_size
patch_size
tubelet_size
tokens_per_frame
embed_dim
dtype
normalization_mean
normalization_std
```

Nếu metadata không khớp, tool không reuse cache và sẽ encode lại. Đây là để tránh lỗi nguy hiểm khi shape vô tình trùng nhưng encoder/checkpoint khác.

### 3.5. Hydra configs mới

File mới:

```text
configs/hydra/experiment/rc_jepa_tiny_mix_oldservo_frame_stride2.yaml
configs/hydra/experiment/rc_jepa_tiny_oldservo_frame_stride2.yaml
```

Cả hai config đều dùng:

```text
predictor_type = simple
model_size = tiny
raw_frames_per_sample = 8
frame_stride = 2
target_fps = 0.0
auto_steps = 2
batch_size = 32
eval_batch_size = 2
warmup_epochs = 5
early_stopping_patience = 15
```

Đây là bản tiny để thử nghiệm nhanh. Nếu kết quả ổn thì mới nâng lên `small` hoặc `base`.

## 4. Kết quả build dataset

### 4.1. Mixed experiment

Lệnh đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python3 -m tools.build_servo_experiment_dataset \
  --mode mixed \
  --experiment-root data/experiments/servo_old_mix_v1 \
  --no-test-split
```

Kết quả manifest:

```text
train samples: 114579
val samples:   49226
test samples:  49226  # test alias val
```

Session split:

```text
current_servo: train 70, val 30, test 30  # test alias val
old_servo:     train 19, val 9,  test 9   # test alias val
```

Sequence thật với cấu hình `raw_frames_per_sample=8`, `frame_stride=2`:

```text
train windows: 79765
  current_servo: 54216
  old_servo:     25549

val windows: 34926
  current_servo: 26061
  old_servo:     8865

test windows: 34926  # test alias val
```

### 4.2. Old-only experiment

Lệnh đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python3 -m tools.build_servo_experiment_dataset \
  --mode old-only \
  --experiment-root data/experiments/servo_old_only_v1 \
  --no-test-split
```

Kết quả manifest:

```text
train samples: 36564
val samples:   12695
test samples:  12695  # test alias val
```

Session split:

```text
old_servo: train 19, val 9, test 9  # test alias val
```

Sequence thật với cấu hình `raw_frames_per_sample=8`, `frame_stride=2`:

```text
train windows: 25549
val windows:   8865
test windows:  8865  # test alias val
```

## 5. Lệnh extract feature

GPU hiện tại đang lỗi driver:

```text
nvidia-smi: failed because it could not communicate with NVIDIA driver
torch.cuda.is_available(): False
```

Vì vậy chưa chạy extract feature thật cho experiment này trong lượt triển khai. Không nên chạy ViT-B feature extraction bằng CPU vì rất chậm và không đúng điều kiện train.

Sau khi GPU hoạt động lại, chạy mixed feature extraction:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --output-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp32 \
  --seed-from-features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --batch-size 32 \
  --dtype fp32 \
  --splits train val
```

Lệnh này sẽ:

- reuse feature của 100 session hiện tại nếu metadata khớp
- encode thêm 28 session servo cũ
- ghi feature cache riêng vào `data/experiments/servo_old_mix_v1/features/...`

Extract old-only:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_only_v1/processed/manifests \
  --output-dir data/experiments/servo_old_only_v1/features/vjepa2_1_vitb_384_ema_fp32 \
  --batch-size 32 \
  --dtype fp32 \
  --splits train val
```

## 6. Lệnh train

### 6.1. Mixed servo tiny

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2
```

Config này chạy:

```text
features_dir = data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp32
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
output_dir = checkpoints/rc_jepa_ac_vitb_features_servo_old_mix_tiny_frame_stride2
```

### 6.2. Old-only tiny

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_oldservo_frame_stride2
```

Config này chạy:

```text
features_dir = data/experiments/servo_old_only_v1/features/vjepa2_1_vitb_384_ema_fp32
manifest_dir = data/experiments/servo_old_only_v1/processed/manifests
output_dir = checkpoints/rc_jepa_ac_vitb_features_oldservo_tiny_frame_stride2
```

## 7. Có cần train lại từ đầu không?

Có. Với experiment này nên train lại từ đầu.

Lý do:

- Data distribution đổi vì thêm servo cũ.
- Split train/val đổi, và test split độc lập bị bỏ khỏi train experiment. `test.jsonl` chỉ còn là alias của `val.jsonl`.
- `frame_stride=2` làm mỗi sample không còn là 8 frame sát nhau.
- Normalization state/action được tính lại từ train manifest.
- Nếu resume checkpoint cũ, optimizer/scheduler và best-val history không còn đại diện cho experiment mới.

Nếu chỉ muốn fine-tune từ checkpoint cũ thì nên làm thành experiment riêng sau, không trộn với kết quả train fresh.

## 8. Rủi ro khi dùng servo cũ

Data servo cũ không chắc cùng phân phối với servo hiện tại.

Các rủi ro chính:

- Servo khác có response khác, cùng `steering_cmd_t` có thể tạo quỹ đạo khác.
- Throttle range của data cũ hẹp hơn, ví dụ report old-only cho thấy `throttle_cmd_t` khoảng `[-0.1, 0.09]`.
- Một số state như `steering_last_t`, `throttle_last_t`, `v_t` có thể phải đọc/suy từ sensor hoặc previous action tùy session.
- Nếu trộn không kiểm soát, model có thể học trung bình giữa hai domain và làm giảm chất lượng trên servo hiện tại.

Vì vậy thứ tự nên chạy:

1. `old-only tiny`: kiểm tra data cũ có học được dynamics hữu hạn không.
2. `mixed tiny`: kiểm tra thêm data cũ có làm `val/domain/current_servo/*` xấu đi không.
3. Chỉ nếu tiny ổn mới chạy `small/base`.

## 9. Kết quả kiểm tra đã chạy

Các kiểm tra đã pass:

```text
py_compile:
  src/tools/extract_vjepa_features.py
  src/tools/build_servo_experiment_dataset.py
  src/tools/train_rc_jepa_ac_features.py
  src/data/feature_sequence_dataset.py
  src/data/sequence_dataset.py

build mixed dataset: pass
build old-only dataset: pass
manifest audit mixed: pass
manifest audit old-only: pass
preprocess_report bad_sessions: 0
Hydra dry-run mixed tiny: pass
Hydra dry-run old-only tiny: pass
```

Kiểm tra chưa chạy được:

```text
Feature extraction thật: chưa chạy do GPU driver lỗi.
Train thật: chưa chạy vì feature experiment chưa extract xong và GPU chưa hoạt động.
```
