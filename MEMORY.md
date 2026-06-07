# MEMORY - Dự án NN-JEPA cho xe RC

## Mục tiêu của dự án

Dự án này đang xây pipeline dữ liệu và model kiểu JEPA/V-JEPA cho xe RC tự lái trong nhà.

Hướng hiện tại:

```text
dùng encoder V-JEPA 2.1 đã pretrained
freeze encoder
train action-conditioned predictor/world model trên dữ liệu xe RC
```

Điểm quan trọng:

- Không sửa code gốc trong `vjepa2/`.
- Folder `vjepa2/` chỉ dùng để tham khảo code public và import encoder V-JEPA 2.1.
- Code của xe RC nằm ngoài `vjepa2/`, chủ yếu trong `src/`.
- Bản hiện tại là world model, chưa phải planner/MPC hoàn chỉnh để tự chọn action.

## Cấu trúc dữ liệu raw

Raw data dự kiến nằm trong:

```text
data/raw/session_xxx/
  frames/
  actions.csv
  telemetry.csv
  accel.csv
  gyro.csv
  rotvec.csv
  gps.csv
  meta.json
```

Ý nghĩa các file:

- `frames/`: ảnh camera đã ghi.
- `actions.csv`: lệnh điều khiển theo frame hoặc theo thời gian.
- `telemetry.csv`: dữ liệu telemetry từ xe, gồm steering/throttle thực tế nếu có.
- `accel.csv`: dữ liệu gia tốc kế.
- `gyro.csv`: dữ liệu con quay hồi chuyển.
- `rotvec.csv`: dữ liệu rotation vector, hiện chưa dùng trong AC model đầu tiên.
- `gps.csv`: dữ liệu GPS, hiện chỉ có thể dùng cho `v_t` nếu đủ ổn định.
- `meta.json`: metadata của session.

## Cấu trúc dữ liệu processed

Processed data nằm trong:

```text
data/processed/
  images/
  manifests/train.jsonl
  manifests/val.jsonl
  manifests/test.jsonl
  reports/preprocess_report.json
```

Mỗi dòng trong manifest là một frame sample:

```python
{
    "sample_id": "...",
    "session_id": "...",
    "frame_index": int,
    "timestamp_sec": float | None,
    "frame_path": "...",
    "source_frame_path": "...",
    "state": {...},
    "action": {...},
    "meta": {...},
}
```

Manifest được split theo `session_id`, không split ngẫu nhiên từng frame, để tránh leakage giữa train/val/test.

## State và action ban đầu

State đầy đủ theo pipeline hiện tại:

```python
s_t = [
    v_t,
    yaw_rate_t,
    accel_x_t,
    accel_y_t,
    steering_last_t,
    throttle_last_t,
]
```

Action:

```python
a_t = [
    steering_cmd_t,
    throttle_cmd_t,
]
```

Ý nghĩa từng biến:

- `v_t`: vận tốc tuyến tính hiện tại của xe tại thời điểm `t`.
- `yaw_rate_t`: tốc độ quay quanh trục z, lấy từ gyro `gz`.
- `accel_x_t`: gia tốc theo trục x, lấy từ accel `ax`.
- `accel_y_t`: gia tốc theo trục y, lấy từ accel `ay`.
- `steering_last_t`: lệnh lái trước đó.
- `throttle_last_t`: lệnh ga trước đó.
- `steering_cmd_t`: lệnh lái tại thời điểm `t`.
- `throttle_cmd_t`: lệnh ga tại thời điểm `t`.

## `v_t` là gì và có tác dụng gì?

`v_t` là vận tốc hiện tại của xe.

Nó có tác dụng quan trọng trong world model vì cùng một action nhưng trạng thái xe khác nhau sẽ tạo chuyển động tương lai khác nhau.

Ví dụ:

```text
Xe đang đứng yên + throttle 0.3 -> bắt đầu tăng tốc
Xe đang chạy nhanh + throttle 0.3 -> tiếp tục đi nhanh hơn hoặc giữ tốc
Xe đang chạy nhanh + steering 0.5 -> cua gắt hơn xe đang chạy chậm
```

Vì vậy, nếu có `v_t` đáng tin cậy, model sẽ dự đoán latent frame tương lai tốt hơn.

Tuy nhiên bản AC đầu tiên đang bỏ `v_t` vì nguồn hiện tại là `gps.speed`. Xe chạy trong nhà thì GPS thường yếu, nhiễu hoặc mất tín hiệu. Nếu đưa tín hiệu nhiễu vào model, model có thể học sai dynamics.

Khi nào nên thêm lại `v_t`:

- Có wheel encoder.
- Có optical flow speed ổn định.
- Có visual odometry.
- GPS hoạt động tốt ở môi trường test.

## State đang dùng cho AC world model

Bản AC world model đầu tiên dùng:

```python
AC_STATE_COLUMNS = [
    yaw_rate_t,
    accel_x_t,
    accel_y_t,
    steering_last_t,
    throttle_last_t,
]
```

Tức là bỏ `v_t`.

Action vẫn là:

```python
AC_ACTION_COLUMNS = [
    steering_cmd_t,
    throttle_cmd_t,
]
```

## `auto_steps` là gì?

`auto_steps` là số bước rollout tự hồi quy dùng trong loss khi train world model.

Nó không phải số frame trong sample.

Nó không phải `tubelet_size`.

Nó không phải số lần encoder chạy.

Nó là số bước model phải tự dùng latent dự đoán của chính nó để dự đoán tiếp.

Ví dụ sample có `T = 8` frame:

```text
frames:  f0 f1 f2 f3 f4 f5 f6 f7
actions: a0 a1 a2 a3 a4 a5 a6
states:  s0 s1 s2 s3 s4 s5 s6 s7
```

Teacher forcing loss học:

```text
latent(f0) + a0 + s0 -> latent(f1)
latent(f1) + a1 + s1 -> latent(f2)
latent(f2) + a2 + s2 -> latent(f3)
...
latent(f6) + a6 + s6 -> latent(f7)
```

Rollout loss với `auto_steps = 2` học thêm:

```text
Bước 1:
latent(f0) thật + a0 + s0 -> dự đoán latent(f1)

Bước 2:
latent(f1) dự đoán + a1 + s1 -> dự đoán latent(f2)
```

Mục đích của `auto_steps`:

- Giúp predictor ổn hơn khi rollout nhiều bước.
- Tránh trường hợp model chỉ đúng khi luôn được ăn latent thật.
- Chuẩn bị cho bước sau là planner/MPC, vì planner sẽ phải rollout nhiều action candidate bằng latent do model tự dự đoán.

Mặc định hiện tại:

```python
AC_AUTO_STEPS = 2
```

## `raw_frames_per_sample` và `tubelet_size`

Mặc định hiện tại:

```python
AC_RAW_FRAMES_PER_SAMPLE = 8
AC_TUBELET_SIZE = 2
AC_AUTO_STEPS = 2
```

`raw_frames_per_sample = 8` nghĩa là một sample train lấy 8 frame thật liên tiếp trong cùng một session.

`tubelet_size = 2` là tham số của encoder V-JEPA video. Nếu đưa trực tiếp clip 8 frame vào encoder với `tubelet_size=2`, về mặt temporal patching có thể thành 4 nhóm thời gian.

Nhưng code robot AC public trong `vjepa2/app/vjepa_droid/train.py` không làm như vậy.

Nó encode từng frame riêng bằng cách duplicate frame đó thành pseudo clip 2 frame:

```text
frame_t -> [frame_t, frame_t]
```

Vì vậy:

```text
8 raw frames -> 8 latent frame steps
```

Không phải:

```text
8 raw frames -> 4 latent frame steps
```

Bản RC AC hiện tại làm giống logic này để giữ mapping rõ ràng:

```text
f0 + a0 + s0 -> f1
f1 + a1 + s1 -> f2
...
f6 + a6 + s6 -> f7
```

## Code hiện tại

Global config:

```text
src/data/settings.py
```

Dataset single-step cho baseline behavior cloning:

```text
src/data/dataset.py
```

Dataset sequence cho AC world model:

```text
src/data/sequence_dataset.py
```

Model baseline behavior cloning:

```text
src/models/rc_car_model.py
```

Model AC world model:

```text
src/models/rc_jepa_ac.py
```

Train baseline behavior cloning:

```text
src/tools/train_rc_car.py
```

Train AC world model:

```text
src/tools/train_rc_jepa_ac.py
```

Tests:

```text
tests/test_pipeline_simple.py
tests/test_rc_jepa_ac.py
```

## Dataset sequence cho AC

Dataset mới là `RCJepaACSequenceDataset`.

Input từ manifest frame-level.

Output một item:

```python
images:  [C, T, H, W]
states:  [T, 5]
actions: [T - 1, 2]
```

Sau DataLoader:

```python
images:  [B, C, T, H, W]
states:  [B, T, 5]
actions: [B, T - 1, 2]
```

Window chỉ được tạo trong cùng một `session_id`. Dataset không nối frame giữa hai session khác nhau.

## AC world model hiện tại

Model chính là `RCJepaACWorldModel`.

Nó gồm:

```text
FrozenVJepa21Encoder
SimpleACPredictor
```

Encoder:

- Import từ `vjepa2/app/vjepa_2_1/models/vision_transformer.py`.
- Load checkpoint V-JEPA 2.1.
- Checkpoint key mặc định là `ema_encoder`.
- Freeze toàn bộ parameter.
- Luôn ở `eval()`.
- Chạy trong `torch.no_grad()`.
- Chỉ tạo latent target tokens.

Predictor:

- Là causal transformer nhỏ, dễ đọc hơn predictor public.
- Nhận latent tokens.
- Nhận action token.
- Nhận state token.
- Chỉ predictor được train.

Loss:

```python
loss = teacher_forcing_loss + rollout_loss
```

Checkpoint của AC train chỉ lưu predictor, optimizer, args và metrics. Không lưu lại encoder frozen để tránh checkpoint quá nặng.

## Lệnh train AC

Ví dụ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac \
  --vjepa-checkpoint /path/to/vjepa2_1_checkpoint.pt \
  --vjepa-root vjepa2 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac
```

Các option quan trọng:

```bash
--checkpoint-key ema_encoder
--encoder vit_base_384
--raw-frames-per-sample 8
--auto-steps 2
--batch-size 8
--epochs 50
--lr 1e-4
```

Nếu checkpoint không có key `ema_encoder`, thử:

```bash
--checkpoint-key target_encoder
```

hoặc:

```bash
--checkpoint-key encoder
```

Nếu checkpoint bị mismatch và cần debug:

```bash
--allow-partial-checkpoint
```

## Kiểm tra đã chạy

Đã chạy:

```bash
python3 -m compileall src tests
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Kết quả pass.

Lưu ý: tại thời điểm chạy test, môi trường shell chưa có `torch`, nên các test tensor/model bị skip. Code mới vẫn import torch trực tiếp, không có fallback import.

## Cập nhật ổn định train mới nhất

Đã sửa 3 vấn đề logic quan trọng:

- Rollout trong `src/models/rc_jepa_ac.py` không còn dùng state tương lai thật. Nó chỉ dùng state ban đầu và action đã biết; riêng `steering_last_t` / `throttle_last_t` được cập nhật từ action trước đó.
- Sequence dataset trong `src/data/sequence_dataset.py` không tạo window nếu frame hoặc timestamp bị đứt quãng.
- Numeric input được normalize bằng thống kê từ train manifest, và checkpoint train lưu lại metadata normalization.

Các biến global mới/cần nhớ trong `src/data/settings.py`:

```python
REMOVE_SIMPLE_OUTLIERS = True
AC_MAX_FRAME_INDEX_GAP = 1
AC_MAX_TIME_GAP_SEC = 0.25
NORMALIZE_STATE_INPUTS = True
NORMALIZE_AC_ACTION_INPUTS = True
NUMERIC_NORMALIZE_CLIP = 8.0
```

Preprocess gần nhất đã rebuild manifest với outlier robust:

```text
train: 29195 samples
val:    6484 samples
test:   13579 samples
```

Report mới:

```text
data/processed/reports/preprocess_report.json
```

## W&B logging

`tools.train_rc_car` và `tools.train_rc_jepa_ac` log lên Weights & Biases mặc định.

Default project:

```text
nn-jepa-rc
```

Metrics:

```text
train/*
val/*
test/*
best/val_loss
lr
```

Tắt W&B:

```bash
--no-wandb
```

Log offline:

```bash
--wandb-mode offline
```

## Quy tắc quan trọng cho các session sau

- Không sửa code trong `vjepa2/`.
- Không dùng fallback import cho torch.
- Không đổi baseline behavior cloning nếu user không yêu cầu.
- Ưu tiên biến global trong `settings.py` để dễ chỉnh.
- Tránh dùng từ `context` trong code RC mới để tránh nhầm với context token của JEPA.
- Dùng `raw_frames_per_sample` để chỉ số frame thật trong một sample train.
- Bản hiện tại là world model, chưa phải planner/MPC.
- World model học dynamics trong latent space, chưa trực tiếp sinh action.

## Việc nên làm tiếp

1. Dùng môi trường có cài `torch`.
2. Chạy preprocess nếu chưa có manifest processed.
3. Chuẩn bị checkpoint V-JEPA 2.1.
4. Chạy smoke train 1 epoch với batch nhỏ.
5. Kiểm tra `run_config.json`, `history.json`, `test_metrics.json`.
6. Xác nhận encoder thật sự freeze và chỉ predictor update.
7. Sau khi loss ổn, thêm planner/MPC hoặc policy head để chọn action.
