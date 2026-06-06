# HANDOFF - NN-JEPA RC Car JEPA-AC

## Mục tiêu hiện tại

Dự án đang xây world model kiểu JEPA/V-JEPA cho xe RC tự lái trong nhà.

Hướng đang làm:

```text
freeze encoder V-JEPA 2.1 đã pretrained
train action-conditioned predictor/world model trên chuỗi dữ liệu xe RC
```

Ràng buộc quan trọng:

- Không sửa bất kỳ file nào trong `vjepa2/`.
- `vjepa2/` chỉ dùng để tham khảo code public và import encoder V-JEPA 2.1.
- Code mới của xe RC nằm trong `src/`.

## Những gì đã implement

Các file mới cho bản AC world model:

```text
src/data/sequence_dataset.py
src/models/rc_jepa_ac.py
src/tools/train_rc_jepa_ac.py
tests/test_rc_jepa_ac.py
```

File config được mở rộng:

```text
src/data/settings.py
```

Các global default mới:

```python
AC_STATE_COLUMNS = (
    "yaw_rate_t",
    "accel_x_t",
    "accel_y_t",
    "steering_last_t",
    "throttle_last_t",
)
AC_ACTION_COLUMNS = ACTION_COLUMNS
AC_RAW_FRAMES_PER_SAMPLE = 8
AC_SEQUENCE_STRIDE = 1
AC_IMAGE_SIZE = 384
AC_TUBELET_SIZE = 2
AC_AUTO_STEPS = 2
```

`v_t` đang bị bỏ khỏi state AC đầu tiên vì nguồn hiện tại là `gps.speed`, mà GPS trong nhà thường nhiễu.

## Giải thích nhanh các khái niệm quan trọng

`v_t` là vận tốc hiện tại của xe tại thời điểm `t`.

Tác dụng của `v_t`: giúp model biết xe đang đứng yên, chạy chậm hay chạy nhanh. Cùng một lệnh steering/throttle nhưng nếu tốc độ khác nhau thì chuyển động tương lai sẽ khác nhau.

Lý do chưa dùng `v_t`: dữ liệu speed hiện tại đến từ GPS, không đáng tin khi chạy trong nhà. Nếu sau này có wheel encoder, optical flow speed hoặc visual odometry thì nên thêm lại.

`auto_steps` là số bước rollout tự hồi quy trong lúc train.

Ví dụ `auto_steps = 2`:

```text
Bước 1: latent thật f0 -> dự đoán latent f1
Bước 2: latent dự đoán f1 -> dự đoán latent f2
```

Nó giúp predictor không chỉ đúng khi ăn latent thật, mà còn ổn khi ăn latent do chính nó dự đoán.

`raw_frames_per_sample = 8` nghĩa là mỗi sample train dùng 8 frame thật liên tiếp.

`tubelet_size = 2` là tham số encoder V-JEPA. Bản RC AC hiện tại làm giống code public `vjepa_droid`: mỗi frame thật được duplicate thành pseudo clip 2 frame trước khi encode.

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

## Data contract

Dataset sequence đọc manifest:

```text
JEPA/data/processed/manifests/train.jsonl
JEPA/data/processed/manifests/val.jsonl
JEPA/data/processed/manifests/test.jsonl
```

Một item dataset trả về:

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

Window chỉ được tạo trong cùng một `session_id`; không nối dữ liệu giữa các session.

## Model contract

Model chính:

```text
RCJepaACWorldModel
```

Nó gồm:

```text
FrozenVJepa21Encoder
SimpleACPredictor
```

Encoder:

- Load từ local `vjepa2/`.
- Dùng checkpoint V-JEPA 2.1.
- Checkpoint key mặc định: `ema_encoder`.
- Freeze toàn bộ parameter.
- Luôn `eval()`.
- Chạy `torch.no_grad()`.
- Chỉ dùng để tạo latent target.

Predictor:

- Causal transformer nhỏ.
- Nhận latent tokens.
- Nhận action token.
- Nhận state token.
- Là phần duy nhất được train.

Loss:

```python
loss = teacher_forcing_loss + rollout_loss
```

Teacher forcing:

```text
latent(f_t) + action_t + state_t -> latent(f_{t+1})
```

Rollout:

```text
dùng latent vừa dự đoán để dự đoán tiếp auto_steps bước
```

## Lệnh train

Ví dụ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac \
  --vjepa-checkpoint /path/to/vjepa2_1_checkpoint.pt \
  --vjepa-root vjepa2 \
  --manifest-dir JEPA/data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac
```

Option thường dùng:

```bash
--checkpoint-key ema_encoder
--encoder vit_base_384
--raw-frames-per-sample 8
--auto-steps 2
--batch-size 8
--epochs 50
--lr 1e-4
```

Nếu checkpoint key mismatch, thử:

```bash
--checkpoint-key target_encoder
```

hoặc:

```bash
--checkpoint-key encoder
```

Nếu cần debug checkpoint mismatch:

```bash
--allow-partial-checkpoint
```

## Trạng thái kiểm tra

Đã chạy:

```bash
python3 -m compileall src tests
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Kết quả pass.

Lưu ý: môi trường shell lúc đó chưa có `torch`, nên test tensor/model bị skip. Code mới vẫn import torch trực tiếp, không có fallback.

## Việc cần làm tiếp

1. Dùng môi trường có `torch`.
2. Chạy preprocessing nếu chưa có manifest.
3. Chuẩn bị checkpoint V-JEPA 2.1.
4. Chạy smoke train 1 epoch với batch nhỏ.
5. Kiểm tra `run_config.json`, `history.json`, `test_metrics.json`.
6. Xác nhận encoder freeze và chỉ predictor update.
7. Sau khi world model loss ổn, thêm planner/MPC hoặc policy head để chọn action.

## Ràng buộc cần nhớ

- Không đụng `vjepa2/`.
- Giữ code đơn giản, dễ đọc, dễ sửa.
- Không thêm fallback import cho torch.
- Không đổi baseline behavior cloning nếu user không yêu cầu.
- Ưu tiên đặt biến dễ chỉnh trong `settings.py`.
- Tránh dùng từ `context` trong code RC mới để không nhầm với context token của JEPA.
