# NN-JEPA

Pipeline dữ liệu này bám đúng repo `JEPA/` hiện có, và đã được cập nhật theo hướng **Android onboard recorder** mới.
Khi session đã chạy `JEPA/src/sync.py`, preprocessing sẽ ưu tiên `actions_synced.csv` + `imu_synced.csv`;
`actions.csv` gốc chỉ còn là fallback cho session cũ/chưa sync.

Luồng hiện tại:

`JEPA/data/raw/session_xxx/{frames, actions_synced.csv, imu_synced.csv, actions.csv, telemetry.csv, accel.csv, gyro.csv, rotvec.csv, gps.csv} -> preprocess -> manifest train/val/test -> Dataset/DataLoader`

Schema model vẫn giữ đúng mục tiêu của bạn:

```text
s_t = [
  v_t,
  yaw_rate_t,
  accel_x_t,
  accel_y_t,
  steering_last_t,
  throttle_last_t
]

a_t = [
  steering_cmd_t,
  throttle_cmd_t
]
```

## Điểm thay đổi chính

- Bỏ YAML + dataclass config nhiều tầng
- Gom toàn bộ biến hay chỉnh vào 1 file: [settings.py](/home/heheboiz/data/NN-JEPA/src/data/settings.py:1)
- Bám trực tiếp format recorder của `JEPA/src/recorder.py`
- Giữ code đủ đơn giản để bạn tự sửa nhanh trong quá trình thu data

## Các file chính

```text
src/data/settings.py      # tất cả biến toàn cục
src/data/preprocess.py    # đọc session, clean, resize, split, write manifest
src/data/dataset.py       # PyTorch Dataset/DataLoader
src/tools/preprocess_data.py
```

## Dữ liệu raw mà pipeline đang hiểu

Mặc định nó đọc session kiểu mới:

```text
JEPA/data/raw/
  session_20260603_120000/
    frames/
      000001.jpg
      000002.jpg
      ...
    actions.csv
    telemetry.csv
    accel.csv
    gyro.csv
    rotvec.csv
    gps.csv
    meta.json
```

`actions.csv` của Android recorder hiện có kiểu:

```csv
frame_idx,t_ms,steering,throttle,seq,esp_ms,mode
1,1234567,0.10,0.20,10,123456,1
2,1234667,0.12,0.22,11,123476,1
```

Các stream phụ hiện có:

```text
telemetry.csv -> t_ms,seq,esp_ms,steering,throttle,mode
accel.csv     -> t_ms,ax,ay,az
gyro.csv      -> t_ms,gx,gy,gz
rotvec.csv    -> t_ms,rx,ry,rz
gps.csv       -> t_ms,lat,lon,alt,speed,bearing,acc
```

Pipeline sẽ ưu tiên `actions_synced.csv` làm mốc theo frame và trộn `imu_synced.csv` theo `frame_idx`.
Nếu chưa có 2 file này, nó fallback sang `actions.csv` rồi ghép các stream phụ theo timestamp gần nhất. Mapping hiện tại:

- `steering/throttle` ở `actions_synced.csv` (ưu tiên) hoặc `telemetry.csv` / `actions.csv` -> `steering_cmd_t`, `throttle_cmd_t`
- `gyro.csv.gz` -> `yaw_rate_t`
- `accel.csv.ax` -> `accel_x_t`
- `accel.csv.ay` -> `accel_y_t`
- `gps.csv.speed` -> `v_t`
- `steering_last_t`, `throttle_last_t` -> lấy từ action trước đó nếu CSV chưa có
- nếu sensor nào thiếu hoặc lệch thời gian quá xa thì điền `MISSING_STATE_VALUE`

## Cảnh báo quan trọng

Với bản Android mới, state vector của bạn đã khá hơn trước, nhưng vẫn có một lưu ý lớn:

- `a_t` là dữ liệu thật
- `yaw_rate_t`, `accel_x_t`, `accel_y_t` có thể lấy từ IMU điện thoại
- `steering_last_t`, `throttle_last_t` vẫn được suy từ action trước
- `v_t` hiện đang lấy từ `gps.csv.speed`, nên với indoor thì chất lượng có thể không tốt

Tức là:

- ngoài trời: `gps speed` có thể tạm dùng
- trong nhà: `v_t` vẫn là biến yếu nhất, nếu muốn chuẩn thì cần wheel odometry / encoder / tốc độ suy tốt hơn

## Chỗ chỉnh nhanh nhất

Mọi thứ cần hay đổi đều nằm ở [settings.py](/home/heheboiz/data/NN-JEPA/src/data/settings.py:1), ví dụ:

- đường dẫn data: `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`
- kích thước ảnh: `IMAGE_WIDTH`, `IMAGE_HEIGHT`
- split: `TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
- stride lấy mẫu: `USE_EVERY_NTH_FRAME`
- có cho phép session chỉ có action hay không: `ALLOW_ACTIONS_ONLY_SESSIONS`
- giá trị điền cho state thiếu: `MISSING_STATE_VALUE`
- ngưỡng match giữa frame và sensor: `TELEMETRY_MATCH_TOL_MS`, `ACCEL_MATCH_TOL_MS`, `GYRO_MATCH_TOL_MS`, `GPS_MATCH_TOL_MS`
- scale action: `STEERING_SCALE`, `THROTTLE_SCALE`
- augmentation: `BRIGHTNESS_JITTER`, `CONTRAST_JITTER`, `HORIZONTAL_FLIP_PROB`

## Chạy preprocessing

Nên sync session trước:

```bash
python3 JEPA/src/sync.py
```

Rồi mới preprocess:

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

Kết quả sẽ nằm ở:

```text
JEPA/data/processed/
  images/
  manifests/
    train.jsonl
    val.jsonl
    test.jsonl
  reports/
    preprocess_report.json
```

## Dùng DataLoader

```python
from data import create_dataloaders

dataloaders = create_dataloaders(batch_size=32, num_workers=4)
batch = next(iter(dataloaders["train"]))

print(batch["image"].shape)
print(batch["state"].shape)
print(batch["action"].shape)
```

## Train baseline

File mới:

```text
src/models/rc_car_model.py
src/tools/train_rc_car.py
```

Baseline này nhận:

- `image`
- `state = [v_t, yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]`

và dự đoán:

- `action = [steering_cmd_t, throttle_cmd_t]`

Chạy bản đơn giản nhất:

```bash
PYTHONPATH=src python3 -m tools.train_rc_car
```

Nếu muốn đổi backbone ảnh sang V-JEPA 2.1 base và load checkpoint local:

```bash
PYTHONPATH=src python3 -m tools.train_rc_car \
  --backbone vjepa2_1_vitb \
  --vjepa-checkpoint /duong/dan/toi/checkpoint.pt \
  --freeze-image-encoder
```

Model được viết theo kiểu dễ sửa:

- `small_cnn`: CNN nhỏ để chạy baseline nhanh
- `vjepa2_1_vitb`, `vjepa2_1_vitl`: bọc encoder local từ repo `vjepa2/`
- `--sensor-names ...`: chọn sensor nào thực sự muốn dùng
- output nằm ở `checkpoints/rc_car_bc/`

## Khi bạn muốn mở rộng

Nếu sắp tới firmware/ESP32 hoặc logger của bạn ghi thêm tốt hơn:

- `v_t`
- `yaw_rate_t`
- `accel_x_t`
- `accel_y_t`
- `steering_last_t`
- `throttle_last_t`

thì chỉ cần giữ đúng tên cột đó trong row frame hoặc stream sensor tương ứng, pipeline sẽ dùng trực tiếp.
