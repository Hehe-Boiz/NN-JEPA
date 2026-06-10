# Báo cáo triển khai experiment trộn data servo cũ

Ngày cập nhật: 2026-06-10

Phạm vi: repo `NN-JEPA`. Repo `JEPA/` chỉ được dùng làm nguồn data/staging, không sửa code gốc trong `JEPA/` hoặc `vjepa2/`.

## 1. Trạng thái hiện tại

Experiment dùng data servo cũ hiện chỉ giữ một nhánh chính:

```text
data/experiments/servo_old_mix_v1
```

Ý nghĩa:

- `servo_old_mix_v1`: trộn data servo hiện tại trong `data/raw` với data servo cũ.
- `servo_old_only_v1`: đã xóa, không còn Hydra config riêng.
- Không copy data cũ vào `data/raw`.
- Không ghi đè `data/processed` baseline.
- Không dùng lại checkpoint train cũ để resume, vì data và split đã đổi.

Feature cache mặc định hiện chuyển sang `fp16`:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
```

Feature cache `fp32` baseline cũ đã bị xóa vì quá nặng:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

## 2. Vì sao bỏ `servo_old_only_v1`

Ban đầu có ý tưởng chạy hai experiment:

```text
mixed    = current servo + old servo
old-only = chỉ old servo
```

Sau khi thống nhất lại, file split chính thức là:

```text
data/split_vjepa_ac_car.json
```

File này chứa cả session current servo và session servo cũ. Vì vậy nó chỉ phù hợp với experiment trộn:

```text
servo_old_mix_v1
```

Nếu dùng split này cho `old-only`, các session current trong split sẽ không có trong source old-servo, dẫn tới split sai mục tiêu. Do đó `old-only` bị bỏ để tránh nhầm:

- Không còn folder `data/experiments/servo_old_only_v1`.
- Không còn config `rc_jepa_tiny_oldservo_frame_stride2.yaml`.
- `tools.build_servo_experiment_dataset` không còn nhận `--mode old-only`.

## 3. Hai kiểu experiment còn dùng

### 3.1. Không trộn data servo cũ

Dùng baseline current data:

```text
manifest_dir = data/processed/manifests
features_dir = data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

Hydra config nên dùng để test nhanh:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata
```

Config này không đọc data servo cũ.

### 3.2. Trộn data servo cũ

Dùng experiment root riêng:

```text
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
features_dir = data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
split_file   = data/split_vjepa_ac_car.json
```

Hydra config nên dùng để test nhanh:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2
```

Config này dùng `frame_stride=2`, gần tinh thần temporal spacing của source V-JEPA AC hơn so với lấy 8 frame sát nhau.

## 4. Nguồn data

Current servo:

```text
data/raw/session_*
```

Old servo:

```text
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV/session_*
```

Audit sau khi cập nhật:

```text
current servo trong data/raw: 181 session
old servo source: 30 session
tổng session trong split_vjepa_ac_car.json: 211 session
missing session so với source hiện có: 0
actions_synced.csv + imu_synced.csv old-servo: 30/30
```

Trong 30 session old servo:

- 28 session có alias `_kds` trong split.
- 2 session old servo được lấy từ `JEPA/data/drive_zips/trong nhà`: `session_20260605_225028`, `session_20260605_225326`.

## 5. Code chính đã thêm/sửa

### 5.1. `src/tools/build_servo_experiment_dataset.py`

Tool này build experiment root riêng.

Mode còn dùng:

```text
mixed        = data/raw + data servo cũ
current-only = chỉ data/raw nhưng ghi ra experiment root riêng
```

Tham số quan trọng:

```bash
--split-file data/split_vjepa_ac_car.json
--no-test-split
```

Khi có `--split-file`, tool không chia random nữa mà dùng đúng danh sách `train` và `val` trong JSON.

Tool cũng tạo alias cho old-servo:

```text
session_xxx
session_xxx_kds
```

Nhờ vậy split file có thể chứa tên old-servo dạng `_kds`, còn folder source thật vẫn là `session_xxx`.

### 5.2. `src/data/sequence_dataset.py`

Dataset sequence từ ảnh processed trả thêm:

```python
"data_domain": first_sample.get("data_domain", "unknown")
```

### 5.3. `src/data/feature_sequence_dataset.py`

Dataset sequence từ feature cache cũng trả thêm `data_domain`.

Điểm này giúp train loop log loss theo domain:

```text
val/domain/current_servo/loss
val/domain/old_servo/loss
```

### 5.4. `src/tools/train_rc_jepa_ac_features.py`

Train loop feature-cache hiện:

- Train loss tổng như cũ.
- Val loss tổng như cũ.
- Val loss theo domain nếu batch có `data_domain`.
- Early stopping và best checkpoint vẫn dựa trên `val/loss` tổng.

### 5.5. `src/tools/extract_vjepa_features.py`

Feature extractor có:

```bash
--seed-from-features-dir
--dtype fp16
```

`--seed-from-features-dir` dùng để reuse feature cache baseline nếu metadata khớp. Với mixed experiment, sau khi baseline fp16 đã extract xong, có thể seed current sessions từ baseline rồi chỉ encode thêm old sessions.

Metadata phải khớp các trường quan trọng:

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

Nếu không khớp, tool sẽ không reuse để tránh trộn feature sai encoder/dtype.

## 6. Kết quả build `servo_old_mix_v1`

Lệnh đã chạy:

```bash
PYTHONPATH=src python3 -m tools.build_servo_experiment_dataset \
  --mode mixed \
  --experiment-root data/experiments/servo_old_mix_v1 \
  --split-file data/split_vjepa_ac_car.json \
  --no-test-split
```

Kết quả:

```text
train samples: 182558
val samples:   30282
test samples:  30282  # test alias val
```

Session split:

```text
train sessions: 169
val sessions:   42
test sessions:  42  # test alias val
```

Domain session split:

```text
current_servo: train 143, val 38, test 38
old_servo:     train 26,  val 4,  test 4
```

Sequence/window count với `raw_frames_per_sample=8`, `frame_stride=2`:

```text
train windows: 128566
val windows:    20501
test windows:   20501  # test alias val
```

Domain sample count:

```text
train current_servo samples: 133677
train old_servo samples:      48881
val current_servo samples:    27360
val old_servo samples:         2922
```

## 7. Lệnh extract feature fp16

### 7.1. Baseline current data

Chạy trước để tạo cache current-data fp16:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --batch-size 32 \
  --dtype fp16
```

Output mặc định:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

### 7.2. Mixed data với seed từ baseline

Chạy sau khi baseline fp16 đã có:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --output-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --seed-from-features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 32 \
  --dtype fp16 \
  --splits train val
```

Nếu baseline fp16 chưa có, bỏ `--seed-from-features-dir` hoặc extract baseline trước.

## 8. Lệnh train

### 8.1. Train tiny không trộn old-servo

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata
```

### 8.2. Train tiny mixed old-servo

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2
```

Nếu muốn xem config trước khi chạy thật:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2 \
  runtime.dry_run=true \
  runtime.require_cuda=false
```

## 9. Có cần train lại từ đầu không?

Có, với mixed experiment nên train lại từ đầu.

Lý do:

- Data distribution đổi vì thêm old-servo.
- Split train/val đổi theo `data/split_vjepa_ac_car.json`.
- `frame_stride=2` làm một sample không còn là 8 frame sát nhau.
- Normalization state/action được tính lại từ train manifest.
- Nếu resume checkpoint cũ, optimizer/scheduler và best-val history không còn đại diện cho experiment mới.

Nếu muốn fine-tune từ checkpoint cũ, nên tạo experiment riêng, không trộn với kết quả train fresh.

## 10. Rủi ro khi trộn servo cũ

Old-servo không chắc cùng phân phối với current-servo.

Rủi ro chính:

- Cùng `steering_cmd_t` có thể tạo góc lái thật khác do servo khác.
- Cùng `throttle_cmd_t` có thể tạo gia tốc khác nếu ESC/motor/pin khác.
- State như `steering_last_t`, `throttle_last_t`, `yaw_rate_t`, `accel_x_t`, `accel_y_t` có thể có nhiễu hoặc bias khác.
- Nếu data cũ nhiều nhưng lệch domain, model có thể giảm chất lượng trên current-servo.

Vì vậy khi train mixed, phải theo dõi cả:

```text
val/loss
val/domain/current_servo/loss
val/domain/old_servo/loss
```

Nếu `val/domain/current_servo/loss` xấu đi nhiều, cần giảm tỷ trọng old-servo hoặc tách domain conditioning rõ hơn.

## 11. Kiểm tra đã chạy

Các kiểm tra sau đã pass sau cập nhật 2026-06-10:

```text
split_vjepa_ac_car.json:
  ids: 211
  unique_ids: 211
  raw_dirs matched: 181
  old_dirs matched: 30
  missing: 0

servo_old_mix_v1:
  train samples: 182558
  val samples: 30282
  test samples: 30282
  train windows: 128566
  val windows: 20501
  test windows: 20501

deleted:
  data/experiments/servo_old_only_v1
  data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

## 12. Cập nhật sau khi sync nốt 2 session old-servo

Ngày cập nhật: 2026-06-10.

Hai session old-servo trước đó còn fallback về `actions.csv`:

```text
session_20260605_225028
session_20260605_225326
```

Đã chạy JEPA sensor sync cho cả hai session:

```text
session_20260605_225028: kept 1156, dropped 1, +imu
session_20260605_225326: kept 1448, dropped 1, +imu
```

Sau đó đã rebuild `servo_old_mix_v1` bằng split file cũ và audit lại:

```text
action_source_counts:
  actions_synced.csv: 211
fallback actions.csv: 0

session_20260605_225028:
  raw_rows: 1156
  kept_rows: 1121
  merged_synced_imu_rows: 1156
  missing_synced_imu_rows: 0

session_20260605_225326:
  raw_rows: 1448
  kept_rows: 1423
  merged_synced_imu_rows: 1448
  missing_synced_imu_rows: 0
```

Manifest/sample count mới:

```text
train samples: 182558
val samples:   30282
test samples:  30282  # test alias val

train domain samples:
  current_servo: 133677
  old_servo:      48881

val domain samples:
  current_servo: 27360
  old_servo:      2922
```

Feature cache mixed fp16 đã được refresh riêng cho 2 session này. Kết quả extractor:

```text
extracted: 2
skipped_compatible: 209
seeded: 0
```

Audit feature cache sau cùng:

```text
features_dir: data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
session_count: 211
frame_count: 212840
dtype: fp16
tokens_per_frame: 576
embed_dim: 768
missing_count: 0
bad_count: 0
```
