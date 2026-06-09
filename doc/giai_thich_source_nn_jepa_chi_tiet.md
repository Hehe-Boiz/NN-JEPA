# Giải thích source NN-JEPA hiện tại

Tài liệu này giải thích source code hiện tại của repo `NN-JEPA` theo hướng đọc được từ trên xuống dưới: cấu trúc repo, luồng dữ liệu, từng module, từng class/hàm chính, input-output, side effect và các điểm dễ gây lỗi. Phạm vi chính là `src/` và `configs/hydra/`. Không giải thích chi tiết code trong repo tham chiếu `JEPA/` và `vjepa2/`, vì hai thư mục đó là nguồn tham khảo/hardware/upstream riêng.

Với các file rất dài, tài liệu này không copy lại từng dòng code, mà gom các dòng liên tục có cùng trách nhiệm thành từng cụm. Cách đọc đúng là mở file code cạnh tài liệu này, rồi lần theo tên file, line range, class/hàm và phần giải thích.

## 1. Bức tranh tổng thể

NN-JEPA hiện tại có 5 phần lớn:

1. Đồng bộ và chuẩn hóa data thô.
2. Tiền xử lý session thành manifest train/val/test.
3. Trích xuất feature V-JEPA 2.1 frozen encoder ra `.npy`.
4. Train predictor/world model trên feature cache.
5. Eval, inference, planning, visualize, web UI.

Luồng chính đang dùng để train feature-cache:

```text
Google Drive / raw session
  -> src/tools/sync_drive_data.py
  -> data/raw/session_...
  -> src/data/preprocess.py hoặc src/tools/preprocess_data.py
  -> data/processed/manifests/train.jsonl, val.jsonl, test.jsonl
  -> src/tools/extract_vjepa_features.py
  -> data/processed/features/<preset>/metadata.json + sessions/*.npy + sessions/*.json
  -> src/tools/train_rc_jepa_ac_features.py
  -> checkpoints/<run>/last_train.pt, last.pt, best.pt, epochs/*.pt, history.json, run_config.json
  -> src/tools/eval_rc_jepa_ac_features.py
  -> src/tools/infer_rc_jepa_ac_features.py
  -> src/tools/plan_rc_jepa_ac_features.py
  -> src/tools/plot_rc_jepa_planning.py
```

Ý tưởng model hiện tại:

```text
frame ảnh
  -> V-JEPA 2.1 frozen encoder
  -> latent tokens mỗi frame
  -> predictor nhận latent quá khứ + state + action
  -> dự đoán latent frame kế tiếp
  -> loss = teacher_forcing_loss + rollout_loss
```

State/action hiện đang dùng:

```python
s_t = [
    yaw_rate_t,
    accel_x_t,
    accel_y_t,
    steering_last_t,
    throttle_last_t,
]

a_t = [
    steering_cmd_t,
    throttle_cmd_t,
]
```

Điểm rất quan trọng: pipeline feature-cache hiện tại không train encoder. Encoder V-JEPA 2.1 chỉ dùng để trích latent trước, sau đó train chỉ cập nhật predictor.

## 2. Cấu trúc thư mục source

```text
src/
  data/
    settings.py
    preprocess.py
    dataset.py
    sequence_dataset.py
    feature_sequence_dataset.py
    normalization.py
    __init__.py
  models/
    rc_car_model.py
    rc_jepa_ac.py
    vjepa21_presets.py
    __init__.py
  tools/
    sync_drive_data.py
    preprocess_data.py
    extract_vjepa_features.py
    train_rc_jepa_ac_features.py
    train_rc_jepa_ac_features_hydra.py
    train_rc_jepa_ac.py
    eval_rc_jepa_ac_features.py
    infer_rc_jepa_ac_features.py
    plan_rc_jepa_ac_features.py
    plot_rc_jepa_planning.py
    rc_jepa_ac_feature_runtime.py
    rc_jepa_ac_cem_planner.py
    session_web_viewer.py
    export_session_gif.py
    progress.py
    wandb_utils.py
    train_rc_car.py
    __init__.py
  viewer/
    index.html
    app.js
    styles.css
configs/
  hydra/
    config.yaml
    experiment/*.yaml
```

Các thư mục không nên đọc như source chính:

```text
src/*.egg-info/
src/**/__pycache__/
outputs/
wandb/
data/
checkpoints/
JEPA/
vjepa2/
```

## 3. Cấu hình Hydra

### 3.1. `configs/hydra/config.yaml`

File này là entrypoint mặc định cho train bằng Hydra.

```yaml
defaults:
  - experiment: rc_jepa_tiny
  - _self_
```

Ý nghĩa:

- Nếu chạy `PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra` mà không override gì, Hydra chọn experiment mặc định là `rc_jepa_tiny`.
- `_self_` nghĩa là config hiện tại được merge sau phần defaults.

```yaml
hydra:
  job:
    chdir: false
```

Ý nghĩa:

- Hydra thường đổi working directory sang thư mục output riêng.
- `chdir: false` giữ nguyên cwd là root repo `NN-JEPA`.
- Đây là quyết định đúng cho repo này vì path data/checkpoint đang viết dạng tương đối như `data/processed/...`.

```yaml
hydra:
  run:
    dir: outputs/hydra/${now:%Y%m%d_%H%M%S}
  output_subdir: null
```

Ý nghĩa:

- Log Hydra vẫn có thư mục output theo thời gian.
- `output_subdir: null` giảm việc tạo thêm `.hydra` phụ.

### 3.2. Cấu trúc chung của `configs/hydra/experiment/*.yaml`

Mỗi experiment thường có các block:

```yaml
output_dir: ...

runtime:
  dry_run: false
  require_cuda: true

data:
  features_dir: ...
  manifest_dir: ...
  state_columns: [...]
  action_columns: [...]
  raw_frames_per_sample: 8
  sequence_stride: 1
  auto_steps: 2

model:
  type: simple hoặc official_lite
  size: tiny/small/base
  predictor_dim: null
  predictor_depth: null
  predictor_heads: null
  dropout: 0.0

train:
  epochs: ...
  batch_size: ...
  eval_batch_size: ...
  num_workers: ...
  lr: ...
  weight_decay: ...
  grad_clip: ...
  warmup_epochs: ...
  min_lr_ratio: ...
  early_stopping_patience: ...
  resume_from: null

wandb:
  disabled: false
  project: nn-jepa-rc
  entity: null
  run_id: null
  continue_run: true
  resume: allow
```

Các key quan trọng:

- `output_dir`: nơi lưu checkpoint.
- `data.features_dir`: nơi chứa feature cache `.npy`.
- `data.manifest_dir`: nơi chứa `train.jsonl`, `val.jsonl`, `test.jsonl`.
- `data.raw_frames_per_sample`: số frame trong một sample train.
- `data.sequence_stride`: bước trượt cửa sổ sample, không phải khoảng cách giữa các frame bên trong sample.
- `data.auto_steps`: số bước rollout autoregressive.
- `model.type`: chọn predictor `simple` hoặc `official_lite`.
- `model.size`: chọn preset kích thước predictor.
- `train.batch_size`: batch train.
- `train.eval_batch_size`: batch val/test. Hiện nên để nhỏ hơn train vì eval dễ OOM do transformer eval fastpath/memory pattern.
- `wandb.continue_run`: bật/tắt resume cùng một W&B run.
- `wandb.run_id`: id run W&B muốn resume.

### 3.3. `rc_jepa_tiny.yaml`

Đây là config tiny mặc định.

Đặc điểm:

- `output_dir = checkpoints/rc_jepa_ac_vitb_features_20260607_tiny`.
- `model.type = simple`.
- `model.size = tiny`.
- `epochs = 20`.
- `batch_size = 10`.
- `eval_batch_size = 2`.
- `warmup_epochs = 2`.
- `early_stopping_patience = 5`.

Config này dùng để thử nhanh, debug pipeline, kiểm tra data và W&B.

### 3.4. `rc_jepa_tiny_newdata.yaml`

Mục tiêu giống tiny nhưng dùng output riêng cho data mới. Dùng khi đã kéo/preprocess/extract feature data mới và muốn không ghi đè checkpoint cũ.

### 3.5. `rc_jepa_small.yaml`

Config trung gian:

- Model lớn hơn tiny.
- Chạy chậm hơn tiny.
- Ít rủi ro OOM hơn base/official-lite.
- Hợp lý nếu tiny chạy ổn nhưng loss/prediction chưa đủ tốt.

### 3.6. `rc_jepa_base.yaml`

Config base:

- `model.type = simple`.
- `model.size = base`.
- Predictor khoảng 20M tham số tùy `tokens_per_frame`, `embed_dim`, state/action dim.
- Đây là bản trước đó bạn train lâu, khoảng gần 1 tiếng/epoch tùy máy/data.

### 3.7. `rc_jepa_base_newdata.yaml`

Giống base nhưng output/checkpoint hướng tới data mới. Dùng để train nghiêm túc sau khi tiny/small ổn.

### 3.8. `rc_jepa_official_lite_tiny.yaml`

Config dùng predictor `official_lite`.

Đặc điểm:

- `model.type = official_lite`.
- `model.size = tiny`.
- `batch_size = 4`.
- `eval_batch_size = 1`.

Vì sao eval batch nhỏ:

- `official_lite` dùng attention mask kiểu V-JEPA AC.
- Sequence rất dài: `T * (2 + tokens_per_frame)`.
- Với ViT-B 384, `tokens_per_frame = 576`, `T = 8`, sequence khoảng `8 * 578 = 4624` token.
- Attention full sequence có chi phí bộ nhớ rất lớn, nhất là khi eval.

## 4. Module `src/data`

Module này chịu trách nhiệm định nghĩa path, đọc raw data, tiền xử lý, tạo manifest, tạo dataset và dataloader.

### 4.1. `src/data/settings.py`

Vai trò:

- Chứa cấu hình toàn cục dễ sửa.
- Gom path, tên file, cột CSV, tham số ảnh, augmentation, split, dataloader, JEPA-AC.
- Các tool/dataset import file này để dùng default.

Nhóm dòng đầu file:

- Import `Path`.
- Xác định `ROOT_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `MANIFEST_DIR`.
- Các path này là nền cho toàn pipeline.

Nhóm cấu hình session/file:

- Tên thư mục frame raw.
- Tên file action CSV.
- Tên file IMU synced.
- Tên file manifest.
- Tên file report.

Nhóm cấu hình ảnh:

- `IMAGE_WIDTH`, `IMAGE_HEIGHT`: kích thước ảnh processed cho pipeline ảnh cũ.
- `AC_IMAGE_SIZE`: kích thước encoder V-JEPA AC, hiện là 384.
- Cần phân biệt: ảnh processed có thể là 224, nhưng khi extract feature, encoder có thể interpolate lên 384.

Nhóm normalization ảnh:

- `NORMALIZE_MEAN = (0.485, 0.456, 0.406)`.
- `NORMALIZE_STD = (0.229, 0.224, 0.225)`.
- Đây là chuẩn ImageNet.

Nhóm action/state:

```python
ACTION_COLUMNS = ("steering_cmd_t", "throttle_cmd_t")
AC_STATE_COLUMNS = (
    "yaw_rate_t",
    "accel_x_t",
    "accel_y_t",
    "steering_last_t",
    "throttle_last_t",
)
AC_ACTION_COLUMNS = ACTION_COLUMNS
```

Ý nghĩa:

- `ACTION_COLUMNS` là action target/control.
- `AC_STATE_COLUMNS` là state input cho JEPA-AC predictor.
- `steering_last_t`, `throttle_last_t` là lệnh điều khiển gần nhất đã gửi trước đó, dùng như một phần trạng thái.

Nhóm split:

- Tỉ lệ train/val/test.
- Seed.
- Split theo session để tránh leakage cùng session qua train và val.

Nhóm DataLoader:

- `BATCH_SIZE`.
- `AC_EVAL_BATCH_SIZE`.
- `NUM_WORKERS`.
- `PIN_MEMORY`.
- `PERSISTENT_WORKERS`.
- `PREFETCH_FACTOR`.
- `SHUFFLE_TRAIN`.

Ảnh hưởng:

- `num_workers` tăng có thể giúp GPU bớt rảnh nhưng tăng RAM.
- `prefetch_factor` tăng làm worker giữ nhiều batch hơn trong RAM.
- `pin_memory=True` tăng tốc copy CPU->GPU nhưng tăng pressure RAM pinned.
- `persistent_workers=True` giữ worker sống qua epoch, nhanh hơn nhưng giữ RAM lâu hơn.

Nhóm JEPA-AC:

- `AC_RAW_FRAMES_PER_SAMPLE = 8`: mỗi sample train gồm 8 frame.
- `AC_SEQUENCE_STRIDE = 1`: cửa sổ sample trượt từng frame một.
- `AC_TUBELET_SIZE = 2`: dùng khi FrozenVJepa21Encoder tạo pseudo clip cho encoder.
- `AC_AUTO_STEPS = 2`: rollout autoregressive 2 bước.
- `AC_MAX_FRAME_INDEX_GAP = 1`: các frame trong window phải liên tiếp theo frame index.
- `AC_MAX_TIME_GAP_SEC = 0.25`: chặn gap timestamp quá lớn.

Điểm dễ nhầm:

- `AC_SEQUENCE_STRIDE` không phải `frame_stride` giữa các frame trong sample.
- Với `raw_frames_per_sample=8`, sample hiện tại là 8 frame liên tiếp nếu data pass `max_frame_index_gap=1`.
- Nếu muốn sample `[0,2,4,6,...]`, cần thêm tham số mới kiểu `frame_stride_inside_sample`, hiện chưa có.

Hàm:

```python
make_output_dirs()
```

Chức năng:

- Tạo các thư mục output cơ bản.
- Không xử lý data.
- Chỉ đảm bảo path tồn tại.

### 4.2. `src/data/__init__.py`

Vai trò:

- Là package initializer cho `data`.
- Có lazy import qua `__getattr__`.

Hàm:

```python
__getattr__(name: str)
```

Chức năng:

- Khi code gọi `from data import X` mà `X` chưa import trực tiếp, hàm này có thể load động.
- Mục tiêu thường là tránh import nặng sớm, nhất là torch/PIL.

### 4.3. `src/data/normalization.py`

File này xử lý chuẩn hóa numeric feature.

Class:

```python
FeatureStats
```

Ý nghĩa:

- Data class lưu `mean` và `std`.
- Mỗi cột state/action có một `FeatureStats`.

Class:

```python
FeatureNormalizer
```

Trách nhiệm:

- Giữ stats cho nhiều cột.
- Normalize từng giá trị/cột.
- Clip giá trị normalized để tránh outlier làm model mất ổn định.

Các method chính:

- `normalize_value`: lấy `(value - mean) / std`, sau đó clip nếu bật.
- `normalize_row`: nhận dict row và danh sách columns, trả list float normalized theo đúng thứ tự columns.

Hàm:

```python
build_feature_normalizer(samples, columns, source_key)
```

Input:

- `samples`: list sample từ manifest.
- `columns`: các cột cần normalize.
- `source_key`: `"state"` hoặc `"action"`.

Output:

- `FeatureNormalizer`.

Logic:

- Duyệt tất cả sample train.
- Lấy value từ `sample[source_key][column]`.
- Tính mean/std.
- Tạo stats cho từng column.

Hàm:

```python
compute_std(values, mean)
```

Chức năng:

- Tính standard deviation.
- Có guard để tránh `std = 0`.

Hàm:

```python
normalizer_to_dict(normalizer)
```

Chức năng:

- Convert normalizer sang dict JSON-serializable để lưu vào checkpoint/run_config.

### 4.4. `src/data/preprocess.py`

Đây là file tiền xử lý raw session thành processed images và manifest.

Hàm chính:

```python
preprocess_all_sessions(progress_callback=None)
```

Vai trò:

- Tìm tất cả session raw.
- Preprocess từng session.
- Gom sample theo session.
- Split session thành train/val/test.
- Ghi `train.jsonl`, `val.jsonl`, `test.jsonl`.
- Ghi report thống kê.

Luồng chính:

1. Gọi `settings.make_output_dirs()`.
2. Gọi `find_session_dirs()`.
3. Với từng session, gọi `preprocess_one_session(session_dir)`.
4. Nếu session không có sample dùng được thì bỏ qua.
5. Nếu không có sample nào sau preprocessing thì raise lỗi.
6. Remove outlier nếu bật trong settings.
7. Split theo session.
8. Ghi manifest JSONL.
9. Tính stats.
10. Trả report.

Dòng lỗi quan trọng:

```python
if not session_samples:
    raise RuntimeError("No usable sample found after preprocessing")
```

Ý nghĩa:

- Sau khi đọc tất cả raw session, không tạo được sample nào hợp lệ.
- Thường do thiếu frame, thiếu CSV, tên file không đúng, action/state không parse được, hoặc range action bị lọc hết.
- Lỗi này giúp fail sớm thay vì train trên dataset rỗng.

Hàm:

```python
find_session_dirs()
```

Chức năng:

- Tìm thư mục session trong `data/raw`.
- Sort tự nhiên theo tên.
- Chỉ trả session dir hợp lệ.

Hàm:

```python
preprocess_one_session(session_dir)
```

Vai trò:

- Xử lý một session.
- Đọc frame.
- Đọc action CSV.
- Đọc IMU/sensor phụ nếu có.
- Match action/state theo frame/timestamp.
- Resize ảnh processed.
- Tạo sample dict.

Một sample manifest thường chứa:

```json
{
  "sample_id": "...",
  "session_id": "...",
  "frame_index": 123,
  "timestamp_sec": 123.456,
  "frame_path": "data/processed/frames/...",
  "raw_frame_path": "data/raw/session.../frames/...",
  "action": {
    "steering_cmd_t": 0.1,
    "throttle_cmd_t": 0.05
  },
  "state": {
    "yaw_rate_t": ...,
    "accel_x_t": ...,
    "accel_y_t": ...,
    "steering_last_t": ...,
    "throttle_last_t": ...
  }
}
```

Hàm:

```python
find_actions_csv_file(session_dir)
```

Chức năng:

- Ưu tiên tìm file action đã sync.
- Nếu không có, fallback sang CSV action raw theo cấu hình.
- Nếu thiếu thì raise `FileNotFoundError`.

Hàm:

```python
merge_synced_imu_rows(action_rows, imu_rows)
```

Chức năng:

- Merge IMU đã sync vào action rows.
- Dùng khi có `imu_synced.csv`.
- Mục tiêu là mỗi action/frame row có thêm sensor như yaw_rate, accel.

Hàm:

```python
load_aux_streams(session_dir)
```

Chức năng:

- Đọc các stream phụ ngoài action CSV.
- Ví dụ IMU/sensor CSV riêng.
- Chuẩn bị time index để match theo timestamp.

Hàm:

```python
build_frame_map(frames_dir)
```

Chức năng:

- Tạo mapping `frame_index -> image_path`.
- Dùng tên file frame để suy ra index.

Hàm:

```python
read_csv_rows(csv_path)
```

Chức năng:

- Đọc CSV thành list dict.
- Mỗi dict là một row.

Hàm:

```python
get_frame_index(row, row_number)
```

Chức năng:

- Cố đọc frame index trong row.
- Nếu không có thì có thể fallback theo row number.

Hàm:

```python
read_frame_index_from_row(row)
```

Chức năng:

- Thử nhiều key có thể chứa frame index.
- Return `int` hoặc `None`.

Hàm:

```python
read_timestamp(row)
read_timestamp_ms(row)
```

Chức năng:

- Đọc timestamp từ CSV.
- Hỗ trợ giây hoặc mili giây.
- Chuẩn hóa về seconds để manifest dùng thống nhất.

Hàm:

```python
read_action(row)
```

Chức năng:

- Đọc `steering_cmd_t`, `throttle_cmd_t`.
- Có thể đọc từ nhiều alias column khác nhau tùy CSV.
- Return dict action hoặc `None` nếu thiếu.

Hàm:

```python
read_state(row, aux_rows=None)
```

Chức năng:

- Đọc state từ row hiện tại và row sensor phụ.
- State gồm yaw_rate, accel_x, accel_y, steering_last, throttle_last.
- Nếu không có `steering_last/throttle_last`, có thể suy từ command hiện tại/cũ tùy logic row.

Hàm:

```python
keep_meta_fields(row)
```

Chức năng:

- Giữ lại metadata từ CSV để debug.
- Không dùng trực tiếp cho train.

Hàm:

```python
action_in_valid_range(action)
```

Chức năng:

- Check action nằm trong range hợp lệ.
- Dùng để bỏ row hỏng/out-of-range.

Hàm:

```python
prepare_image(source_path, output_path)
```

Chức năng:

- Mở ảnh raw.
- Convert RGB.
- Resize về `settings.IMAGE_WIDTH x settings.IMAGE_HEIGHT`.
- Lưu ảnh processed.

Cảnh báo quan trọng:

- Hiện processed image thường là 224x224.
- Feature extractor đọc `frame_path` từ manifest, tức ảnh processed.
- Encoder V-JEPA 2.1 sau đó interpolate lên `AC_IMAGE_SIZE=384` nếu cần.
- Nghĩa là nếu không đổi pipeline, feature V-JEPA 384 có thể đang được extract từ ảnh processed 224 rồi upscale lên 384, không phải từ raw 384.

Hàm:

```python
remove_simple_outliers(samples)
```

Chức năng:

- Bỏ outlier đơn giản theo numeric feature/action.
- Thường dùng median và ngưỡng trong settings.
- Return `(clean_samples, removed_count)`.

Hàm:

```python
median(values)
```

Chức năng:

- Tính median thủ công.

Hàm:

```python
build_session_split(session_ids)
```

Chức năng:

- Split theo session id.
- Output mapping `session_id -> split`.
- Đây là đúng để tránh cùng session xuất hiện cả train và val/test.

Hàm:

```python
compute_feature_stats(session_samples)
```

Chức năng:

- Tính thống kê data sau preprocess.
- Dùng cho report/audit.

Hàm:

```python
natural_sort_key(path)
extract_digits(text)
```

Chức năng:

- Sort tên session/frame theo số tự nhiên.
- Ví dụ `frame_2` đứng trước `frame_10`.

Hàm:

```python
compute_std(values)
```

Chức năng:

- Tính std cho report preprocess.

Hàm:

```python
read_first_float(row, keys)
```

Chức năng:

- Thử đọc float theo nhiều tên cột.
- Return float đầu tiên parse được.

Hàm:

```python
read_state_value(...)
```

Chức năng:

- Đọc từng field state với alias/fallback.

Hàm:

```python
uses_synced_action_row(row)
```

Chức năng:

- Kiểm tra row có phải format synced không.

Hàm:

```python
build_time_index(rows)
```

Chức năng:

- Tạo cấu trúc để tìm row gần timestamp nhanh hơn.

Hàm:

```python
match_sensor_rows(...)
nearest_row(...)
```

Chức năng:

- Match sensor phụ theo timestamp gần nhất.
- Có ngưỡng delta để tránh match sai thời gian.

### 4.5. `src/data/dataset.py`

Đây là dataset cũ cho supervised/image pipeline, không phải feature-cache JEPA-AC chính, nhưng vẫn còn trong repo.

Class:

```python
TrainAugmentor
```

Chức năng:

- Augmentation cho ảnh đơn.
- Hỗ trợ flip, brightness, contrast, saturation, blur.
- Nếu flip ngang, các cột liên quan steering/yaw có thể đổi dấu qua `HORIZONTAL_FLIP_SIGN_COLUMNS`.

Class:

```python
DrivingJEPADataset
```

Vai trò:

- Đọc manifest từng frame.
- Load ảnh.
- Convert tensor.
- Normalize ảnh.
- Trả sample cho model supervised/old baseline.

Output thường có:

```python
{
    "image": tensor[C,H,W],
    "state": tensor[S],
    "action": tensor[A],
    "sample_id": ...,
}
```

Hàm:

```python
load_manifest(path)
```

Chức năng:

- Đọc JSONL manifest.
- Mỗi line là một sample dict.
- Return list sample.

Hàm:

```python
image_to_tensor(image)
```

Chức năng:

- PIL image -> torch tensor float `[C,H,W]`.
- Scale pixel về `[0,1]`.

Hàm:

```python
normalize_tensor(tensor, mean, std)
```

Chức năng:

- Normalize theo ImageNet hoặc mean/std truyền vào.

Hàm:

```python
create_dataloaders(...)
```

Chức năng:

- Tạo DataLoader train/val/test cho dataset ảnh cũ.

Hàm:

```python
_require_numpy()
```

Chức năng:

- Guard import numpy cho code path cần numpy.

### 4.6. `src/data/sequence_dataset.py`

Đây là dataset sequence từ ảnh processed, dùng cho train JEPA-AC trực tiếp với encoder trong graph. Hiện pipeline chính thường dùng `feature_sequence_dataset.py`, nhưng file này vẫn rất quan trọng vì cung cấp window logic dùng chung.

Class:

```python
SequenceAugmentor
```

Chức năng:

- Augment đồng nhất trên tất cả frame trong một sequence.
- Nếu brightness factor được chọn, áp cùng factor cho toàn sequence.
- Điều này tránh làm video sequence bị nhiễu augmentation khác nhau từng frame.

Class:

```python
RCJepaACSequenceDataset
```

Input:

- `split`.
- `manifest_path`.
- `raw_frames_per_sample`.
- `sequence_stride`.
- `state_columns`.
- `action_columns`.
- normalizer.

`__init__` làm gì:

1. Validate `raw_frames_per_sample >= 2`.
2. Validate `sequence_stride >= 1`.
3. Load manifest.
4. Gọi `build_sequence_windows`.
5. Tạo augmentor nếu split train.

`__len__`:

- Return số sequence window.

`__getitem__`:

1. Lấy danh sách sample indices của window.
2. Load từng ảnh từ `frame_path`.
3. Lấy state cho đủ `T` frame.
4. Lấy action cho `T-1` bước.
5. Augment nếu train.
6. Convert ảnh sang tensor và normalize.
7. Stack thành `images_tensor` shape `[C,T,H,W]`.
8. Normalize state/action nếu có normalizer.
9. Return dict.

Output:

```python
{
    "images": torch.Tensor[C,T,H,W],
    "states": torch.Tensor[T,S],
    "actions": torch.Tensor[T-1,A],
    "sample_id": "...",
    "session_id": "...",
    "frame_indices": torch.Tensor[T],
    "timestamps_sec": torch.Tensor[T],
}
```

Hàm:

```python
build_sequence_windows(samples, raw_frames_per_sample, sequence_stride, ...)
```

Đây là hàm cực kỳ quan trọng.

Logic:

1. Gom sample theo `session_id`.
2. Chỉ giữ sample có đủ state/action columns.
3. Sort sample trong mỗi session theo timestamp rồi frame index.
4. Tạo sliding window có độ dài `raw_frames_per_sample`.
5. Bước trượt window là `sequence_stride`.
6. Kiểm tra window liên tục bằng `is_contiguous_window`.

Ví dụ hiện tại:

```text
raw_frames_per_sample = 8
sequence_stride = 1
window 1 = frame [0,1,2,3,4,5,6,7]
window 2 = frame [1,2,3,4,5,6,7,8]
window 3 = frame [2,3,4,5,6,7,8,9]
```

Nếu `sequence_stride = 2`:

```text
window 1 = frame [0,1,2,3,4,5,6,7]
window 2 = frame [2,3,4,5,6,7,8,9]
```

Điểm dễ nhầm:

- `sequence_stride` chỉ đổi khoảng cách giữa hai sample/window.
- Nó không biến frame bên trong sample thành `[0,2,4,6,...]`.

Hàm:

```python
has_required_columns(sample, state_columns, action_columns)
```

Chức năng:

- Check sample có đủ state/action dict và đủ cột cần dùng.

Hàm:

```python
sample_sort_key(sample)
```

Chức năng:

- Sort bằng timestamp trước, frame index sau.
- Nếu timestamp invalid thì dùng `inf`.

Hàm:

```python
is_contiguous_window(samples, window, max_frame_index_gap, max_time_gap_sec)
```

Chức năng:

- Kiểm tra các frame trong window có tăng đúng thứ tự.
- Nếu frame gap > ngưỡng thì bỏ.
- Nếu timestamp gap <=0 hoặc quá lớn thì bỏ.

Hàm:

```python
timestamp_to_float(value)
```

Chức năng:

- Convert timestamp sang float.
- Nếu lỗi trả `nan`.

Hàm:

```python
create_ac_sequence_dataloaders(...)
```

Chức năng:

- Tạo DataLoader train/val/test cho sequence ảnh.
- Build normalizer từ train samples.
- Train shuffle theo `settings.SHUFFLE_TRAIN`.
- Val/test không shuffle.

Hàm:

```python
build_ac_action_normalizer(train_samples, action_columns, state_normalizer)
```

Chức năng:

- Build action normalizer.
- Có logic đặc biệt map:
  - `steering_cmd_t` dùng stats của `steering_last_t`.
  - `throttle_cmd_t` dùng stats của `throttle_last_t`.
- Mục đích: action command và last command cùng loại đại lượng nên nên normalize cùng scale.

### 4.7. `src/data/feature_sequence_dataset.py`

Đây là dataset chính cho train hiện tại từ feature cache.

Constants:

```python
FEATURE_METADATA_NAME = "metadata.json"
FEATURE_SESSIONS_DIR_NAME = "sessions"
```

Ý nghĩa:

- Feature cache có format:

```text
features_dir/
  metadata.json
  sessions/
    session_x.npy
    session_x.json
```

Class:

```python
RCJepaACFeatureSequenceDataset
```

Vai trò:

- Đọc manifest.
- Build sequence windows giống `RCJepaACSequenceDataset`.
- Nhưng thay vì load ảnh, load latent token từ `.npy` memmap.

`__init__` làm gì:

1. Validate `raw_frames_per_sample`.
2. Validate `sequence_stride`.
3. Load `metadata.json`.
4. Lấy `tokens_per_frame`, `embed_dim`.
5. Load manifest split.
6. Build windows.
7. Xác định session nào được dùng trong windows.
8. Mở `np.load(..., mmap_mode="r")` cho từng session.

Điểm bộ nhớ:

- `.npy` được mở memmap nên không load toàn bộ feature vào RAM ngay.
- Nhưng DataLoader worker + prefetch + copy từng frame vẫn có thể làm RAM tăng.

`__len__`:

- Return số sequence windows.

`__getitem__`:

1. Lấy window.
2. Lấy `sequence` là list sample.
3. Xác định session id.
4. Load feature từng frame bằng `session_features.get_frame(frame_index)`.
5. Stack latent frame shape `[T,K,D]`.
6. Reshape thành `[T*K,D]`.
7. Lấy state `[T,S]`.
8. Lấy action `[T-1,A]`.
9. Normalize nếu có normalizer.
10. Return dict.

Output:

```python
{
    "latents": torch.Tensor[T*K,D],
    "states": torch.Tensor[T,S],
    "actions": torch.Tensor[T-1,A],
    "sample_id": "...",
    "session_id": "...",
    "frame_indices": torch.Tensor[T],
    "timestamps_sec": torch.Tensor[T],
}
```

Với config hiện tại thường là:

```text
T = raw_frames_per_sample = 8
K = tokens_per_frame = 576 nếu ViT-B 384
D = embed_dim = 768 nếu ViT-B 384
latents shape mỗi sample = [4608, 768]
```

Class:

```python
SessionFeatureIndex
```

Vai trò:

- Bọc một session feature array `.npy`.
- Giữ mapping `frame_index -> row`.

Method:

```python
get_frame(frame_index)
```

Logic:

- Nếu frame index không có trong cache thì raise `KeyError`.
- Lấy row tương ứng.
- Copy numpy row bằng `np.array(..., copy=True)`.
- Convert sang torch float32.

Điểm quan trọng:

- Dù feature cache lưu `fp16`, dataset convert về `float32`.
- Hiện bạn chọn `fp32`, nên không đổi precision.

Hàm:

```python
load_feature_metadata(features_dir)
```

Chức năng:

- Đọc `metadata.json`.
- Check có `tokens_per_frame`, `embed_dim`, `dtype`.

Hàm:

```python
load_session_feature_index(features_dir, session_id)
```

Chức năng:

- Mở `sessions/session_id.npy`.
- Đọc `sessions/session_id.json`.
- Build `frame_to_row`.
- Check array phải shape `[N,K,D]`.
- Return `SessionFeatureIndex`.

Hàm:

```python
create_ac_feature_sequence_dataloaders(...)
```

Chức năng:

- Tạo train/val/test DataLoader cho feature-cache.
- Build normalizer từ train manifest.
- Train shuffle.
- Val/test không shuffle.
- Dùng `eval_batch_size` riêng cho val/test.

Điểm OOM/RAM:

- Train batch có thể lớn hơn val batch nhưng vẫn chạy do memory pattern khác nhau.
- Val dùng `torch.no_grad()` nhưng transformer eval fastpath và attention allocation có thể vẫn spike.
- Với official_lite sequence dài, `eval_batch_size=1` hoặc `2` là lựa chọn an toàn.

## 5. Module `src/models`

### 5.1. `src/models/vjepa21_presets.py`

File này định nghĩa danh sách encoder V-JEPA 2.1 có thể chọn khi extract feature.

Class:

```python
VJepa21EncoderSpec
```

Trường thường có:

- `name`: tên encoder nội bộ.
- `builder_name`: tên function build model trong source `vjepa2`.
- `embed_dim`: latent dimension.
- `patch_size`: thường 16.

Class:

```python
VJepa21FeaturePreset
```

Trường thường có:

- `name`: preset như `vitb_384`.
- `encoder_name`: tên encoder truyền vào model.
- `checkpoint_path`: checkpoint mặc định.
- `checkpoint_key`: key lấy state dict trong checkpoint.
- `image_size`: 384.
- `patch_size`: 16.
- `tubelet_size`: 2.
- `description`.

Các preset thường gặp:

- `vitb_384`: ViT-B, feature dim 768, checkpoint key `ema_encoder`.
- `vitl_384`: ViT-L, feature dim lớn hơn.
- `vitg_384`: ViT-g.
- `vitG_384`: ViT-G/gigantic.

Hàm:

```python
get_vjepa21_feature_preset(name)
```

Chức năng:

- Lấy preset theo tên.
- Nếu tên sai thì raise lỗi kèm danh sách preset hợp lệ.

Hàm:

```python
vjepa21_feature_output_dir(preset_name, dtype="fp32")
```

Chức năng:

- Tạo output dir chuẩn cho feature cache theo preset và dtype.
- Ví dụ `data/processed/features/vjepa2_1_vitb_384_ema_fp32`.

Hàm:

```python
vjepa21_feature_preset_options()
```

Chức năng:

- Trả danh sách preset cho web UI chọn.

### 5.2. `src/models/rc_car_model.py`

Đây là model supervised/behavior cloning cũ, không phải world model feature-cache chính.

Hàm:

```python
build_sensor_indices(sensor_names)
```

Chức năng:

- Map tên sensor sang index trong tensor state.

Hàm:

```python
select_sensor_features(state, sensor_indices)
```

Chức năng:

- Chọn subset state theo indices.

Class:

```python
SmallImageEncoder
```

Vai trò:

- CNN nhỏ encode ảnh.
- Dùng cho baseline nhẹ.

Forward:

- Input ảnh `[B,C,H,W]`.
- Output feature vector `[B,D]`.

Class:

```python
VJepa2ImageEncoder
```

Vai trò:

- Wrapper dùng V-JEPA encoder cho ảnh.
- Có freeze option.
- Dùng trong baseline supervised nếu muốn dùng encoder pretrained.

Class:

```python
RCDrivingModel
```

Vai trò:

- Model dự đoán action trực tiếp từ image feature + sensor.

Forward:

- Encode ảnh.
- Chọn sensor.
- Concatenate.
- MLP head dự đoán steering/throttle.

### 5.3. `src/models/rc_jepa_ac.py`

Đây là file model quan trọng nhất cho JEPA-AC.

Constants:

```python
DEFAULT_ENCODER_NAME
DEFAULT_CHECKPOINT_KEY = "ema_encoder"
DEFAULT_PREDICTOR_DIM = 512
DEFAULT_PREDICTOR_DEPTH = 6
DEFAULT_PREDICTOR_HEADS = 8
DEFAULT_PATCH_SIZE = 16
DEFAULT_PREDICTOR_TYPE = "simple"
SUPPORTED_PREDICTOR_TYPES = ("simple", "official_lite")
```

Ý nghĩa:

- Default encoder là V-JEPA 2.1 preset mặc định.
- Checkpoint key mặc định `ema_encoder`, tức lấy encoder EMA pretrained.
- Predictor có 2 biến thể: `simple` và `official_lite`.

Preset size:

```python
PREDICTOR_SIZE_PRESETS = {
    "tiny":  dim 128, depth 2, heads 4,
    "small": dim 256, depth 4, heads 4,
    "base":  dim 512, depth 6, heads 8,
}
```

Hàm:

```python
torch_transformer_eval_fastpath_disabled(disable)
```

Chức năng:

- Tạm tắt native eval fastpath của PyTorch transformer.
- Lý do: eval fastpath đôi khi tạo memory spike với sequence token dài.
- Được dùng trong `SimpleACPredictor.forward` khi eval.

Hàm:

```python
apply_predictor_size_preset(args)
```

Chức năng:

- Nếu `predictor_dim/depth/heads` đang là `None`, điền từ `model_size`.
- Check `predictor_dim % predictor_heads == 0`.
- Nếu sai thì raise sớm.

Hàm:

```python
build_ac_predictor(...)
```

Chức năng:

- Factory tạo predictor theo `predictor_type`.
- `simple` -> `SimpleACPredictor`.
- `official_lite` -> `VJepaStyleACPredictor`.

Class:

```python
FrozenVJepa21Encoder
```

Vai trò:

- Load encoder V-JEPA 2.1 từ repo `vjepa2`.
- Load checkpoint.
- Freeze toàn bộ weight.
- Encode ảnh thành latent tokens.

`__init__` làm gì:

1. Lưu path repo `vjepa2`.
2. Lưu checkpoint path/key.
3. Lưu image size, patch size, tubelet size.
4. Build encoder bằng `_build_encoder`.
5. Load checkpoint nếu có.
6. Set `requires_grad=False`.
7. Set eval mode.

Property:

```python
embed_dim
```

- Return latent dimension của encoder.

Property:

```python
tokens_per_frame
```

- Tính `(image_size // patch_size) ** 2`.
- Với `image_size=384`, `patch_size=16`: `24*24=576`.

Method:

```python
train(mode=True)
```

Chức năng:

- Override để encoder luôn eval.
- Dù model cha gọi `.train()`, encoder vẫn frozen/eval.

Method:

```python
forward(images)
```

Input:

- `images` shape `[B,C,T,H,W]`.

Logic:

1. Validate 5D.
2. Validate channel RGB = 3.
3. Reshape từ `[B,C,T,H,W]` sang frame batch `[B*T,C,H,W]`.
4. Nếu H/W khác image_size thì interpolate.
5. Tạo pseudo clip bằng `frames.unsqueeze(2).repeat(... tubelet_size ...)`.
6. Gọi encoder trong `torch.no_grad()`.
7. LayerNorm output nếu bật.
8. Reshape về `[B,T,K,D]`.
9. Flatten thành `[B,T*K,D]`.
10. Detach để không backprop vào encoder.

Vì sao pseudo clip:

- V-JEPA video encoder cần temporal tubelet.
- Mỗi frame đơn được lặp lại thành clip ngắn độ dài `tubelet_size`.
- Cách này cho ra token đại diện từng frame.

Method:

```python
_build_encoder()
```

Chức năng:

- Check repo `vjepa2` tồn tại.
- Thêm `vjepa2` vào `sys.path`.
- Import builder từ `app.vjepa_2_1.models.vision_transformer`.
- Build encoder theo spec.

Method:

```python
_load_checkpoint(checkpoint_path, strict)
```

Chức năng:

- Load checkpoint bằng torch.
- Lấy state dict qua `extract_checkpoint_state`.
- Clean key.
- Load vào encoder.
- Nếu strict false, in missing/unexpected keys.

Class:

```python
SimpleACPredictor
```

Vai trò:

- Predictor causal transformer đơn giản.
- Nhận latent tokens + action token + state token.
- Dự đoán latent token frame kế tiếp.

`__init__`:

- Lưu dims.
- `cond_tokens = 2` vì mỗi frame có 1 action token và 1 state token.
- `latent_proj`: latent dim -> predictor dim.
- `state_proj`: state dim -> predictor dim.
- `action_proj`: action dim -> predictor dim.
- `frame_pos`: positional embedding theo thời gian.
- `patch_pos`: positional embedding theo patch trong frame.
- `action_type`, `state_type`: embedding phân biệt loại token.
- `nn.TransformerEncoderLayer`.
- `nn.TransformerEncoder`.
- `LayerNorm`.
- `output_proj`: predictor dim -> latent dim.

Forward:

Input:

```python
latent_tokens: [B,T*K,D]
actions: [B,T,A]
states: [B,T,S]
```

Trong train loss, khi teacher forcing:

- `latent_tokens` là frame `0..T-2`.
- `actions` là action `0..T-2`.
- `states` là state `0..T-2`.
- Output phải dự đoán latent frame `1..T-1`.

Logic:

1. Check token/frame.
2. Tính `num_frames = total_tokens // tokens_per_frame`.
3. Check actions/states shape.
4. Reshape latent `[B,T,K,D]`.
5. Project latent/action/state.
6. Add positional/type embeddings.
7. Ghép sequence theo frame:
   ```text
   [action_t, state_t, patch_1_t, ..., patch_K_t]
   ```
8. Flatten thành `[B, T*(K+2), predictor_dim]`.
9. Tạo time causal mask.
10. Chạy transformer.
11. Bỏ condition tokens, chỉ lấy patch tokens.
12. Project về latent dim.

Mask:

- Token ở thời điểm `t` chỉ nhìn được token thời điểm `<=t`.
- Đây là causal theo time.

Class:

```python
VJepaStyleACPredictor
```

Vai trò:

- Predictor official-lite, bám gần source V-JEPA AC public hơn Simple.
- Có action/state tokens trước patch tokens.
- Có action-block causal attention mask.
- Có RoPE attention tự viết.

Khác `SimpleACPredictor`:

- Không dùng `nn.TransformerEncoder`.
- Dùng `VJepaStyleACBlock`.
- Dùng `VJepaStyleACAttention`.
- Có `build_action_block_causal_attention_mask`.
- Cấu trúc giống tinh thần `VisionTransformerPredictorAC`, nhưng chưa y chang Meta vì đã giản lược và điều chỉnh cho RC feature cache.

`__init__`:

1. Check `predictor_dim % num_heads`.
2. Check `tokens_per_frame` là số chính phương.
3. Với 576 token, grid là `24x24`.
4. Tạo encoder linear cho latent/action/state.
5. Tạo nhiều block.
6. Tạo norm/proj.
7. Register attention mask max length.
8. Init weights.
9. Rescale block weights kiểu ViT.

Forward:

1. Check shape.
2. Project latent/action/state.
3. Ghép `[action,state,patches]` mỗi frame.
4. Slice attention mask theo sequence length thực tế.
5. Chạy từng block.
6. Reshape về frame.
7. Bỏ action/state tokens.
8. Project về latent dim.

Class:

```python
VJepaStyleACBlock
```

Vai trò:

- Một transformer block kiểu ViT:
  - norm1
  - attention
  - residual
  - norm2
  - MLP
  - residual

Forward:

```python
x = x + attention(norm1(x))
x = x + mlp(norm2(x))
```

Class:

```python
VJepaStyleACAttention
```

Vai trò:

- Multi-head attention có RoPE 3D/time-height-width cho patch token.
- Action/state token có RoPE theo time.
- Dùng `torch.nn.functional.scaled_dot_product_attention`.

`__init__`:

- Check dim chia hết heads.
- Tạo `qkv`, `proj`.
- Tính số chiều RoPE cho time/height/width.

Forward:

1. Check sequence length đúng layout.
2. Tách action/state token và patch token.
3. QKV action/state riêng.
4. QKV patch token.
5. Apply RoPE cho patch.
6. Merge action/state heads và patch heads.
7. Chạy SDPA với `attn_mask`.
8. Project output.

Method phụ:

- `_qkv`: tạo q/k/v.
- `_encode_action_tokens`: encode action/state token theo time.
- `_apply_patch_rope`: áp RoPE cho frame/height/width.
- `_merge_action_and_patch_heads`: ghép lại token condition và patch.

Class:

```python
VJepaStyleMLP
```

Vai trò:

- MLP trong transformer block.
- `Linear -> GELU -> Dropout -> Linear -> Dropout`.

Class:

```python
RCJepaACWorldModel
```

Vai trò:

- Model full online gồm frozen encoder + predictor.
- Dùng cho `train_rc_jepa_ac.py`, không phải feature-cache trainer chính.

`__init__`:

1. Tạo `FrozenVJepa21Encoder`.
2. Tạo predictor bằng `build_ac_predictor`.

`train`:

- Gọi train cho model cha.
- Ép `target_encoder.eval()`.

`forward`:

1. Encode images thành latents.
2. Gọi `compute_world_model_losses`.

Hàm:

```python
compute_world_model_losses(...)
```

Đây là loss cốt lõi.

Input:

```python
latents: [B,T*K,D]
states: [B,T,S]
actions: [B,T-1,A]
tokens_per_frame: K
auto_steps: số bước rollout
```

Teacher forcing:

```python
input_latents = latents[:, :-K]      # frame 0..T-2
target_latents = latents[:, K:]      # frame 1..T-1
teacher_pred = predictor(input_latents, actions, states[:, :-1])
teacher_forcing_loss = L1(teacher_pred, target_latents)
```

Ý nghĩa:

- Cho model biết latent quá khứ thật.
- Học dự đoán latent kế tiếp.

Rollout:

```python
rollout_tokens = latents[:, :K]      # chỉ frame đầu thật
for step in range(auto_steps):
    pred_tokens = predictor(rollout_tokens, actions[:, :step+1], rollout_states[:, :step+1])
    next_tokens = pred_tokens[:, -K:]
    rollout_tokens = concat(rollout_tokens, next_tokens)
```

Ý nghĩa:

- Sau bước đầu, model phải tự dùng prediction của nó để dự đoán tiếp.
- Đây là dynamics/world model thật hơn teacher forcing.

Loss tổng:

```python
loss = teacher_forcing_loss + rollout_loss
```

Hàm:

```python
build_rollout_state_context(initial_state, actions, rollout_steps, state_columns, action_columns)
```

Chức năng:

- Tạo state input cho rollout mà không dùng future measured state.
- Ban đầu repeat state đầu tiên.
- Sau đó copy previous action vào `steering_last_t`, `throttle_last_t` nếu có cột.

Vì sao:

- Khi inference thật, không thể biết chính xác future state measured.
- Nhưng có thể biết lệnh action vừa gửi, nên update `last command`.

Hàm:

```python
copy_previous_action_to_state(...)
```

Chức năng:

- Copy action ở step trước vào state ở step hiện tại.
- Chỉ làm nếu tên cột tồn tại.

Hàm:

```python
build_time_causal_mask(num_frames, tokens_per_step, device)
```

Chức năng:

- Tạo mask causal đơn giản cho `SimpleACPredictor`.
- Return bool mask kiểu `True = bị mask` theo API TransformerEncoder PyTorch.

Hàm:

```python
build_action_block_causal_attention_mask(num_frames, grid_height, grid_width, add_tokens=2)
```

Chức năng:

- Tạo mask allowed-attention kiểu V-JEPA AC.
- Mỗi frame là một block gồm action/state tokens + patch tokens.
- Query frame `t` được attend source frame `<=t`.
- Return bool mask theo SDPA semantics: `True = allowed`.

Điểm cần phân biệt:

- PyTorch `TransformerEncoder` mask dùng semantics khác với SDPA mask.
- File này xử lý riêng từng predictor nên không được trộn mask hai loại.

Hàm:

```python
rotate_queries_or_keys(x, pos)
```

Chức năng:

- Áp RoPE rotation.
- Yêu cầu chiều cuối chẵn.

Hàm:

```python
expand_source_rope_frequencies(values, target_ndim)
```

Chức năng:

- Broadcast sin/cos RoPE cho đúng số chiều tensor q/k.

Hàm:

```python
separate_patch_positions(token_ids, grid_height, grid_width)
```

Chức năng:

- Từ token id tuyến tính, suy ra:
  - frame id.
  - height id.
  - width id.

Hàm:

```python
extract_checkpoint_state(checkpoint, checkpoint_key)
```

Chức năng:

- Nếu checkpoint là dict có key như `ema_encoder`, lấy key đó.
- Nếu checkpoint bản thân đã là state_dict tensor thì dùng trực tiếp.
- Nếu key thiếu thì raise lỗi kèm keys available.

Hàm:

```python
clean_state_dict_keys(state_dict)
```

Chức năng:

- Bỏ prefix `module.` và `backbone.`.
- Giúp load checkpoint từ DDP/model wrapper.

Hàm:

```python
count_trainable_parameters(model)
```

Chức năng:

- Đếm tham số có `requires_grad=True`.

## 6. Module `src/tools`

### 6.1. `src/tools/preprocess_data.py`

Đây là CLI wrapper rất mỏng quanh `src/data/preprocess.py`.

Hàm:

```python
print_progress(current, total, label)
```

Chức năng:

- In progress khi preprocess.
- Có thể phục vụ web UI/job log.

Hàm:

```python
main()
```

Chức năng:

- Gọi `preprocess_all_sessions`.
- In summary/report.

Cách chạy:

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

### 6.2. `src/tools/sync_drive_data.py`

File này đồng bộ data từ Google Drive bằng `rclone`, giải nén zip session, chạy sensor sync và preprocess nếu được bật.

Hàm:

```python
parse_args()
```

Chức năng:

- Đọc CLI args như remote path, staging dir, raw dir, overwrite, dry-run, skip preprocess.

Hàm:

```python
run_command(command, env=None, dry_run=False)
```

Chức năng:

- Chạy subprocess.
- Nếu dry-run thì chỉ in command.

Hàm:

```python
print_progress(...)
```

Chức năng:

- In progress marker dạng máy đọc được.
- Web UI dùng để hiện thanh tiến trình.

Hàm:

```python
rclone_common_args(args)
```

Chức năng:

- Build phần args chung cho rclone.

Hàm:

```python
sync_zip_staging(args)
```

Chức năng:

- Sync các file `.zip` từ Drive về staging local.
- Mục tiêu tránh tải lại file đã có.

Hàm:

```python
sync_extra_nonzip_staging(args)
```

Chức năng:

- Sync file/thư mục không phải zip nếu có.

Hàm:

```python
check_zip_staging(args)
```

Chức năng:

- Kiểm tra zip staging.
- Có thể so local với remote.

Hàm:

```python
zip_signature(zip_path)
read_signature(session_dir)
write_signature(session_dir, signature)
```

Chức năng:

- Tạo/lưu chữ ký zip.
- Biết session đã giải nén từ zip nào.
- Tránh giải nén lại nếu zip chưa đổi.

Hàm:

```python
move_extracted_session(temp_dir, session_dir, session_name)
```

Chức năng:

- Di chuyển session sau khi giải nén từ temp vào `data/raw`.

Hàm:

```python
extract_session_zip(zip_path, raw_dir, overwrite_changed, dry_run)
```

Chức năng:

- Giải nén một zip session.
- Nếu session đã có và signature khớp thì skip.
- Nếu zip đổi và `overwrite_changed` bật thì overwrite.

Hàm:

```python
extract_top_level_zips(args)
```

Chức năng:

- Duyệt staging, giải nén toàn bộ zip top-level.
- Return summary created/skipped/updated.

Hàm:

```python
session_needs_sensor_sync(session_dir)
```

Chức năng:

- Check session thiếu file sync sensor/action.

Hàm:

```python
run_sensor_sync(args, extract_summary)
```

Chức năng:

- Chạy script sync sensor từ repo `JEPA` nếu cần.
- Mục tiêu tạo `actions_synced.csv`, `imu_synced.csv`.

Hàm:

```python
run_preprocess(dry_run)
```

Chức năng:

- Gọi preprocessing sau khi sync/extract xong nếu không skip.

Hàm:

```python
main()
```

Chức năng:

- Orchestrate toàn bộ sync pipeline.

### 6.3. `src/tools/extract_vjepa_features.py`

Đây là tool trích feature V-JEPA 2.1 ra cache `.npy`.

Class:

```python
FrameFeatureDataset
```

Vai trò:

- Dataset frame đơn cho feature extraction.
- Đọc `frame_path` từ manifest.
- Convert RGB.
- Normalize ImageNet.
- Return tensor ảnh và metadata frame.

Output mỗi item thường có:

```python
{
    "image": tensor[C,H,W],
    "session_id": "...",
    "frame_index": ...,
    "sample": ...
}
```

Hàm:

```python
parse_args()
```

CLI chính:

- `--vjepa-root`.
- `--vjepa-checkpoint`.
- `--encoder`.
- `--checkpoint-key`.
- `--manifest-dir`.
- `--output-dir`.
- `--batch-size`.
- `--num-workers`.
- `--dtype fp32/fp16`.
- `--preset`.

Hàm:

```python
resolve_feature_extraction_args(args)
```

Chức năng:

- Nếu người dùng chọn preset, tự điền encoder/checkpoint/key/output.
- Giúp web UI chọn model dễ hơn.

Hàm:

```python
default_device()
```

Chức năng:

- Chọn `cuda`, `mps`, hoặc `cpu`.

Hàm:

```python
set_seed(seed)
```

Chức năng:

- Set random seed.

Hàm:

```python
print_progress(...)
```

Chức năng:

- In progress marker cho web UI.

Hàm:

```python
load_unique_samples(manifest_dir, splits)
```

Chức năng:

- Đọc train/val/test manifests.
- Lấy unique frame theo session/frame_index.
- Tránh extract lặp cùng frame xuất hiện trong nhiều window.

Hàm:

```python
group_samples_by_session(samples)
```

Chức năng:

- Gom frame theo session.
- Feature được lưu mỗi session một `.npy`.

Hàm:

```python
numpy_dtype(dtype_name)
```

Chức năng:

- Map `"fp32"` -> `np.float32`.
- Map `"fp16"` -> `np.float16`.

Hàm:

```python
build_encoder(args)
```

Chức năng:

- Tạo `FrozenVJepa21Encoder`.
- Load checkpoint.
- Move lên device.

Hàm:

```python
extract_session_features(...)
```

Đây là hàm chính cho từng session.

Logic:

1. Tạo dataset frame cho session.
2. Tạo DataLoader.
3. Tạo memmap output `.npy` shape `[N,K,D]`.
4. Với từng batch:
   - Move image lên device.
   - Tạo shape `[B,C,1,H,W]` cho encoder.
   - Encode.
   - Lưu token về `.npy`.
5. Ghi `.json` index mapping frame_index -> row.

Output:

```text
sessions/session_x.npy
sessions/session_x.json
```

Hàm:

```python
read_existing_metadata(output_dir)
```

Chức năng:

- Đọc metadata cũ nếu có.

Hàm:

```python
metadata_matches_request(existing, requested)
```

Chức năng:

- So metadata cache hiện có với request hiện tại.
- Nếu cùng encoder/checkpoint/dtype/manifest thì có thể skip.

Hàm:

```python
cache_status_from_metadata(...)
cache_status_summary(...)
```

Chức năng:

- Kiểm tra cache thiếu session nào, session nào có rồi.
- Dùng để không extract lại nếu đủ.

Hàm:

```python
write_json(path, payload)
```

Chức năng:

- Ghi JSON pretty.

Hàm:

```python
main()
```

Luồng:

1. Parse args.
2. Resolve preset.
3. Load samples.
4. Build metadata request.
5. Check cache.
6. Nếu cache đủ và metadata khớp thì skip.
7. Build encoder.
8. Extract từng session.
9. Ghi metadata.

Cảnh báo:

- `batch_size` ở extract chỉ ảnh hưởng tốc độ/VRAM, không đổi kết quả nếu cùng dtype/model/input.
- `num_workers` ở extract chỉ ảnh hưởng tốc độ/I/O/RAM.
- `dtype=fp32` chính xác hơn nhưng cache rất lớn.
- `dtype=fp16` nhẹ hơn nhưng có sai số lượng tử hóa feature.

### 6.4. `src/tools/train_rc_jepa_ac_features.py`

Đây là train loop chính hiện tại cho feature cache.

Constants:

```python
DEFAULT_FEATURES_DIR = data/processed/features/vjepa2_1_vitb_384_ema_fp32
DEFAULT_OUTPUT_DIR = checkpoints/rc_jepa_ac_vitb_features_20260607
```

Hàm:

```python
parse_args(argv=None)
```

CLI args chính:

- data: features dir, manifest dir, state/action columns, raw frames, sequence stride, auto steps.
- model: predictor type, model size, dim/depth/heads/dropout.
- train: epochs, batch size, eval batch size, workers, lr, weight decay, grad clip, warmup, min lr ratio, early stopping, resume.
- wandb: project/entity/run id/resume/logging.

Chi tiết quan trọng:

- Gọi `add_wandb_args(parser)` để thêm W&B flags.
- Ghi `_output_dir_was_provided` để biết có nên auto suffix output dir theo predictor type/size không.

Hàm:

```python
default_device()
```

Chức năng:

- Chọn cuda/mps/cpu.

Hàm:

```python
set_seed(seed)
```

Chức năng:

- Seed random và torch.

Hàm:

```python
build_predictor(args, tokens_per_frame, embed_dim)
```

Chức năng:

- Gọi `build_ac_predictor`.
- State dim = số state columns.
- Action dim = số action columns.

Hàm:

```python
run_epoch(...)
```

Đây là loop train/val chung.

Input quan trọng:

- `predictor`.
- `dataloader`.
- `optimizer` hoặc `None`.
- `lr_scheduler` hoặc `None`.
- `tokens_per_frame`.
- `auto_steps`.
- `wandb_run`.

Logic:

1. `training = optimizer is not None`.
2. `predictor.train(training)`.
3. Tạo totals loss.
4. Duyệt batch bằng `tqdm`.
5. Move `latents/states/actions` lên device.
6. Nếu train:
   - lấy current lr.
   - zero grad.
7. Gọi `compute_world_model_losses`.
8. Nếu train:
   - backward.
   - log gradient pre-clip nếu bật.
   - clip grad nếu `grad_clip > 0`.
   - log gradient post-clip nếu bật.
   - optimizer step.
   - scheduler step.
   - log param stats nếu bật.
9. Cộng loss theo batch size.
10. Update tqdm postfix.
11. Log W&B batch metrics theo `wandb_log_every`.

Metrics trong epoch:

- `loss`.
- `teacher_forcing_loss`.
- `rollout_loss`.

Hàm:

```python
average_metrics(totals, total_samples)
```

Chức năng:

- Chia tổng loss weighted theo số sample.

Hàm:

```python
collect_normalization_metadata(dataset)
```

Chức năng:

- Lấy state/action normalizer để lưu checkpoint.

Hàm:

```python
args_to_jsonable_dict(args)
```

Chức năng:

- Convert argparse Namespace sang dict ghi JSON/checkpoint.
- Path -> string.
- Bỏ key private `_...`.

Hàm:

```python
write_json(path, payload)
```

Chức năng:

- Ghi JSON.

Hàm:

```python
epoch_wandb_metrics(...)
```

Chức năng:

- Build metrics log cuối epoch.
- Gồm train/val loss, best val loss, best epoch, early stop patience, lr.

Lưu ý:

- Trong source hiện tại có thể thấy `val` metrics được merge hai lần. Điều này không làm sai kết quả vì key giống nhau ghi đè cùng giá trị, nhưng là dư thừa có thể dọn sau.

Hàm:

```python
save_checkpoint(...)
```

Checkpoint chứa:

- `epoch`.
- `phase`.
- `predictor_state_dict`.
- `optimizer_state_dict`.
- `lr_scheduler_state_dict`.
- `args`.
- `metrics`.
- `state_columns`, `action_columns`.
- `normalization`.
- `feature_metadata`.
- `best_val_loss`.
- `best_epoch`.
- `global_step`.
- `epochs_without_improvement`.
- `history`.
- note: encoder weights không được lưu.

Các file checkpoint:

- `last_train.pt`: lưu sau train xong nhưng trước val.
- `last.pt`: lưu sau val xong.
- `best.pt`: lưu khi val loss tốt nhất.
- `epochs/epoch_XXX.pt`: lưu từng epoch hoàn chỉnh.

Hàm:

```python
load_resume_checkpoint(resume_path, predictor, optimizer, device, args)
```

Chức năng:

- Load checkpoint.
- Validate config predictor.
- Load predictor weights.
- Load optimizer state.

Hàm:

```python
validate_resume_predictor_config(checkpoint, args)
```

Chức năng:

- Check resume không đổi predictor type/dim/depth/heads/dropout.
- Nếu mismatch thì raise lỗi.

Hàm:

```python
maybe_cleanup_cuda()
```

Chức năng:

- Gọi `torch.cuda.empty_cache()` nếu có cuda.
- Dùng trước val/test để giảm fragment.

Hàm:

```python
main(args=None)
```

Luồng chính:

1. Parse args nếu chưa truyền.
2. Apply model size preset.
3. Auto đổi output dir nếu model size/type khác base và user không truyền output dir.
4. Set seed.
5. Tạo output dirs.
6. Build dataloaders.
7. Lấy `tokens_per_frame`, `embed_dim`, `feature_metadata`.
8. Build predictor.
9. Build AdamW optimizer.
10. Tính steps/epoch, total steps, warmup steps.
11. Build normalization metadata.
12. Ghi `run_config.json`.
13. Nếu resume:
    - Load checkpoint.
    - Nếu phase là `train_complete_waiting_val`, resume ngay val của epoch đó.
    - Nếu phase là `epoch_complete`, bắt đầu epoch kế tiếp.
14. Build PyTorch LambdaLR scheduler.
15. Load scheduler state nếu resume.
16. Sync scheduler theo global_step.
17. Init W&B.
18. Watch model nếu bật.
19. Loop epoch:
    - Train epoch.
    - Save `last_train.pt`.
    - Cleanup CUDA.
    - Val epoch trong `torch.no_grad()`.
    - Update history.
    - Update best val.
    - Save `last.pt`, epoch checkpoint, `best.pt` nếu improved.
    - Log W&B epoch.
    - Check early stopping sau warmup.
20. Sau train, load `best.pt`.
21. Test trên test dataloader.
22. Ghi summary/W&B.

Early stopping:

- Dựa trên `val_metrics["loss"]`.
- Warmup epochs không tính vào patience.
- Nếu val không cải thiện sau patience epoch thì stop.

Scheduler:

- Dùng `LambdaLR`.
- Warmup từ `warmup_start_factor * lr` lên `lr`.
- Sau warmup giảm tuyến tính/cosine tùy implementation trong `train_rc_jepa_ac.py` hiện tại.
- `lr_scheduler.step()` chạy mỗi train batch, không chạy theo epoch.

### 6.5. `src/tools/train_rc_jepa_ac_features_hydra.py`

File này là bridge Hydra -> trainer argparse.

Hàm:

```python
build_train_args(cfg)
```

Chức năng:

- Nhận Hydra config.
- Convert sang `argparse.Namespace` giống CLI của `train_rc_jepa_ac_features.py`.
- Map:
  - `data.features_dir` -> `args.features_dir`.
  - `model.type` -> `args.predictor_type`.
  - `model.size` -> `args.model_size`.
  - `train.batch_size` -> `args.batch_size`.
  - `wandb.run_id` -> `args.wandb_run_id`.

Hàm:

```python
to_plain_dict(cfg)
to_plain_value(value)
```

Chức năng:

- Convert OmegaConf/DictConfig/ListConfig thành dict/list/value thường.

Hàm:

```python
require_mapping(root, key)
```

Chức năng:

- Lấy block config bắt buộc.
- Nếu thiếu hoặc sai type thì raise rõ ràng.

Hàm:

```python
path_value(value)
optional_path(value)
optional_int(value)
optional_str(value)
```

Chức năng:

- Convert type an toàn từ config.

Hàm:

```python
hydra_entrypoint(cfg)
```

Chức năng:

- Entry decorator Hydra.
- Build args.
- Nếu `runtime.require_cuda=true` mà không có CUDA thì raise.
- Nếu `runtime.dry_run=true`, in config và không train.
- Nếu không dry-run, gọi trainer `main(args)`.

Hàm:

```python
run()
```

Chức năng:

- CLI entrypoint gọi `hydra_entrypoint`.

Cách chạy tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_tiny
```

Cách chạy official-lite tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_official_lite_tiny
```

### 6.6. `src/tools/train_rc_jepa_ac.py`

Đây là trainer online/raw-image version, tức encoder chạy trong train loop. Hiện feature-cache trainer dùng lại nhiều helper scheduler từ file này.

Hàm:

```python
parse_args(argv=None)
```

CLI cho train online:

- vjepa root.
- checkpoint.
- encoder name.
- checkpoint key.
- model type/size.
- train hyperparameters.

Hàm:

```python
default_device()
set_seed(seed)
```

Giống trainer feature.

Hàm:

```python
build_model(args)
```

Chức năng:

- Tạo `RCJepaACWorldModel`.
- Bao gồm frozen encoder + predictor.

Hàm:

```python
compute_steps_per_epoch(dataloader)
```

Chức năng:

- Return `len(dataloader)`.

Hàm:

```python
compute_warmup_steps(warmup_epochs, steps_per_epoch, total_train_steps)
```

Chức năng:

- Warmup steps = `warmup_epochs * steps_per_epoch`, có guard không vượt total.

Hàm:

```python
should_apply_early_stopping(epoch, warmup_epochs)
```

Chức năng:

- Return true nếu epoch đã qua warmup.

Hàm:

```python
compute_lr_scale(...)
```

Chức năng:

- Tính hệ số lr theo global step.
- Có warmup và decay về `min_lr_ratio`.

Hàm:

```python
build_lr_scheduler(...)
```

Chức năng:

- Tạo `LambdaLR` từ `compute_lr_scale`.

Hàm:

```python
sync_lr_scheduler(lr_scheduler, global_step)
```

Chức năng:

- Khi resume, sync scheduler tới đúng global step.

Hàm:

```python
run_epoch(...)
```

Chức năng:

- Loop train/val cho model online.
- Khác feature trainer ở chỗ batch có `images` và model tự encode.

Hàm:

```python
average_metrics(...)
save_checkpoint(...)
args_to_jsonable_dict(...)
write_json(...)
collect_normalization_metadata(...)
epoch_wandb_metrics(...)
load_resume_checkpoint(...)
validate_resume_predictor_config(...)
main()
```

Chức năng tương tự feature trainer nhưng checkpoint có thể liên quan model full.

### 6.7. `src/tools/eval_rc_jepa_ac_features.py`

Tool eval checkpoint trên split val/test.

Hàm:

```python
parse_args()
```

Args thường có:

- checkpoint path.
- features dir.
- manifest dir.
- split.
- batch size.
- num workers.

Hàm:

```python
maybe_cleanup_cuda()
```

Chức năng:

- Empty cache trước/sau eval.

Hàm:

```python
main()
```

Luồng:

1. Load checkpoint bằng runtime helper.
2. Build predictor.
3. Validate feature metadata.
4. Build dataloaders.
5. Chọn split.
6. Run epoch eval bằng `run_epoch` với `optimizer=None`.
7. In metrics.

### 6.8. `src/tools/infer_rc_jepa_ac_features.py`

Tool inference offline trên feature cache.

Hàm:

```python
parse_args()
```

Args:

- checkpoint.
- features dir.
- manifest dir.
- split.
- batch size.
- output jsonl.

Hàm:

```python
maybe_cleanup_cuda()
```

Chức năng:

- Dọn CUDA cache.

Hàm:

```python
predict_batch(predictor, batch, ...)
```

Chức năng:

- Chạy predictor trên một batch.
- Tạo prediction latent.
- So với target latent.
- Return prediction/target/error.

Hàm:

```python
per_sample_l1(prediction, target)
```

Chức năng:

- Tính L1 cho từng sample.

Hàm:

```python
tensor_to_list(value)
```

Chức năng:

- Convert tensor sang list để ghi JSON.

Hàm:

```python
write_jsonl(path, records)
```

Chức năng:

- Ghi inference records.

Hàm:

```python
main()
```

Luồng:

1. Load checkpoint/model.
2. Build dataloader split.
3. Duyệt batch.
4. Dự đoán latent.
5. Ghi records nếu output path có.
6. In summary.

### 6.9. `src/tools/rc_jepa_ac_feature_runtime.py`

File helper dùng chung cho eval/infer/planning.

Class:

```python
FeaturePredictorConfig
```

Vai trò:

- Gom config predictor lấy từ checkpoint.
- Bao gồm predictor type, dims, columns, tokens_per_frame, embed_dim, auto_steps.

Hàm:

```python
default_device()
```

Chọn CUDA/MPS/CPU.

Hàm:

```python
resolve_checkpoint_path(path)
```

Chức năng:

- Nếu path là folder, có thể tự chọn `best.pt`.
- Nếu path là file, dùng trực tiếp.

Hàm:

```python
load_feature_checkpoint(path, device)
```

Chức năng:

- Load torch checkpoint.
- Return checkpoint dict và path resolved.

Hàm:

```python
config_from_checkpoint(checkpoint)
```

Chức năng:

- Đọc args/metadata trong checkpoint.
- Build `FeaturePredictorConfig`.

Hàm:

```python
build_predictor_from_checkpoint(checkpoint, device)
```

Chức năng:

- Build predictor đúng architecture.
- Load `predictor_state_dict`.
- Move device.
- Set eval.

Hàm:

```python
checkpoint_default_path(checkpoint, key, fallback)
```

Chức năng:

- Lấy path default từ checkpoint args hoặc fallback.

Hàm:

```python
validate_feature_metadata(checkpoint, features_dir)
```

Chức năng:

- So metadata checkpoint với feature cache hiện tại.
- Tránh dùng checkpoint train trên feature layout khác.

### 6.10. `src/tools/rc_jepa_ac_cem_planner.py`

File này triển khai planner CEM offline trên latent feature.

Class:

```python
CEMPlanResult
```

Trường:

- best actions.
- best score.
- candidate scores.
- rollout prediction.

Hàm:

```python
normalizer_stats_tensors(...)
normalize_action_tensor(...)
denormalize_action_tensor(...)
```

Chức năng:

- Convert stats normalize action sang tensor.
- Normalize/denormalize action khi planner sample action trong physical space hoặc normalized space.

Class:

```python
RCJepaACFeatureCEMPlanner
```

Vai trò:

- Dùng predictor để thử nhiều chuỗi action.
- Chọn action sequence làm latent cuối gần goal nhất.

Logic CEM:

1. Khởi tạo distribution action mean/std.
2. Sample nhiều candidate action sequence.
3. Rollout predictor cho từng candidate.
4. Score theo khoảng cách latent prediction tới goal.
5. Chọn elite candidates.
6. Update mean/std.
7. Lặp nhiều iteration.
8. Return best sequence.

Đây hiện là offline planner, chưa phải closed-loop điều khiển xe thật.

### 6.11. `src/tools/plan_rc_jepa_ac_features.py`

CLI dùng CEM planner trên dataset feature-cache.

Hàm:

```python
parse_args()
```

Args:

- checkpoint.
- features dir.
- split.
- horizon.
- goal offset.
- candidates.
- iterations.
- action bounds.
- output jsonl/csv.

Hàm:

```python
set_seed(seed)
```

Seed planner.

Hàm:

```python
resolve_horizon(value, auto_steps, raw_frames_per_sample)
```

Chức năng:

- Xác định planning horizon hợp lệ.
- Không vượt quá số frame sample có thể so target.

Hàm:

```python
resolve_goal_offset(value, horizon, raw_frames_per_sample)
```

Chức năng:

- Xác định frame goal trong sample.

Hàm:

```python
resolve_action_bounds(...)
```

Chức năng:

- Xác định steering/throttle min-max.

Hàm:

```python
tensor_to_list(value)
mean_or_none(values)
write_jsonl(path, records)
write_csv(path, records, action_columns)
```

Chức năng:

- Helper output.

Hàm:

```python
score_final_l1(predictions, goal_tokens)
```

Chức năng:

- Score prediction cuối bằng L1 với latent goal.

Hàm:

```python
main()
```

Luồng:

1. Load checkpoint/model.
2. Build dataset.
3. Với mỗi sample, lấy context/goal.
4. Chạy CEM.
5. Ghi action plan và metrics.

### 6.12. `src/tools/plot_rc_jepa_planning.py`

Tool vẽ chart SVG từ output planning.

Hàm:

```python
parse_args()
read_jsonl(path)
infer_action_columns(records)
get_float(record, key)
collect_series(records, keys)
safe_domain(values)
```

Chức năng:

- Đọc records.
- Tìm action columns.
- Gom series.
- Tính domain chart an toàn.

Hàm:

```python
make_svg_line_chart(...)
```

Chức năng:

- Tạo SVG line chart bằng string.
- Không phụ thuộc matplotlib.

Hàm:

```python
action_sequence_series(...)
```

Chức năng:

- Gom chuỗi action theo horizon.

Hàm:

```python
write_text(path, content)
write_summary(path, output_files, records)
```

Chức năng:

- Ghi SVG/HTML summary.

Hàm:

```python
main()
```

Chức năng:

- Đọc planning output.
- Ghi chart files.

### 6.13. `src/tools/export_session_gif.py`

Tool tạo GIF xem nhanh session.

Hàm:

```python
parse_args()
```

Args:

- session id.
- source raw/processed.
- every nth frame.
- max frames.
- width.
- fps.
- output.

Hàm:

```python
resolve_frames_dir(session_id, source)
```

Chức năng:

- Tìm thư mục frame raw hoặc processed.

Hàm:

```python
collect_frame_paths(frames_dir, every_nth_frame, max_frames)
```

Chức năng:

- Lấy danh sách frame path theo sampling.

Hàm:

```python
resize_image(image, width)
```

Chức năng:

- Resize giữ tỉ lệ.

Hàm:

```python
export_gif(frame_paths, output_path, width, fps)
```

Chức năng:

- Ghi GIF.

Hàm:

```python
default_output_path(session_id)
main()
```

Chức năng:

- Chọn output mặc định và chạy export.

### 6.14. `src/tools/progress.py`

File nhỏ cho progress text.

Class:

```python
ProgressBar
```

Vai trò:

- In progress bar đơn giản trong terminal.

Hàm:

```python
format_metrics(metrics)
```

Chức năng:

- Format dict metrics thành string ngắn.

### 6.15. `src/tools/wandb_utils.py`

File helper W&B.

Hàm:

```python
add_wandb_args(parser)
```

Thêm args:

- `--no-wandb`.
- `--wandb-project`.
- `--wandb-entity`.
- `--wandb-run-name`.
- `--wandb-run-id`.
- `--wandb-resume`.
- `--wandb-mode`.
- `--wandb-tags`.
- `--wandb-log-every`.
- `--wandb-watch-log`.
- `--wandb-watch-freq`.
- `--wandb-grad-stats-every`.
- `--wandb-param-stats-every`.
- `--wandb-continue-run` hoặc config tương đương trong Hydra.

Hàm:

```python
init_wandb(args, config, job_type)
```

Chức năng:

- Nếu W&B disabled thì return None.
- Import wandb.
- Resolve run id.
- Gọi `wandb.init`.

Hàm:

```python
wandb_run_id_path(args)
read_saved_wandb_run_id(args)
resolve_wandb_run_id(args)
should_continue_wandb_run(args)
persist_wandb_run_id(args, run, job_type)
```

Chức năng:

- Quản lý resume cùng W&B run.
- Lưu run id local để lần sau tiếp tục nếu config cho phép.

Hàm:

```python
log_metrics(run, metrics, step=None)
update_summary(run, values)
finish_wandb(run)
flatten_metrics(prefix, metrics)
watch_model(run, model, args)
```

Chức năng:

- Log metric.
- Update summary.
- Finish run.
- Prefix metric name như `train/loss`.
- Watch gradients/parameters nếu bật.

Hàm:

```python
collect_gradient_metrics(model, prefix="grad")
collect_parameter_metrics(model, prefix="param")
collect_tensor_metrics(...)
metric_group_name(name)
```

Chức năng:

- Tính thống kê tensor:
  - global L2.
  - max abs.
  - mean abs.
  - nonfinite count.
  - tensor count.
  - value count.
- Dùng để log gradient/parameter lên W&B.

### 6.16. `src/tools/session_web_viewer.py`

Đây là web server local để xem session và chạy job sync/preprocess/extract feature từ UI.

Hàm:

```python
parse_args()
```

Args:

- host.
- port.
- source.

Hàm:

```python
get_source_root(source)
get_frames_dir(source, session_id)
list_frame_paths(frames_dir)
```

Chức năng:

- Resolve raw/processed source.
- Lấy danh sách frame cho session.

Hàm:

```python
collect_sessions(source)
```

Chức năng:

- Quét session.
- Return metadata để UI hiển thị.

Hàm:

```python
describe_session_files(source, session_dir)
```

Chức năng:

- Liệt kê file quan trọng trong session.

Hàm:

```python
build_session_payload(source, session_id)
```

Chức năng:

- Tạo JSON payload cho một session:
  - frames.
  - file list.
  - session info.

Hàm:

```python
read_static_file(name)
```

Chức năng:

- Đọc `src/viewer/index.html`, `app.js`, `styles.css`.

Hàm:

```python
build_sync_command()
build_preprocess_command()
build_extract_feature_command(...)
build_job_command(job_name, payload=None)
```

Chức năng:

- Build command cho background job:
  - sync data.
  - preprocess.
  - extract feature.

Hàm:

```python
initial_progress(job_name)
completed_progress(status)
parse_progress_line(line)
```

Chức năng:

- Chuẩn hóa progress state cho UI.
- Parse line kiểu `__JOB_PROGRESS__`.

Class:

```python
JobRunner
```

Vai trò:

- Quản lý một background process.
- Start job.
- Read stdout/stderr.
- Update status/progress.
- Cho API query trạng thái.

Class:

```python
SessionViewerHandler(BaseHTTPRequestHandler)
```

Vai trò:

- HTTP request handler.

Các endpoint thường có:

- Serve `/`.
- Serve static JS/CSS.
- API list sessions.
- API get session frames.
- API start job.
- API get job status.
- API serve image frame.

Hàm:

```python
query_value(query, key, default=None)
main()
```

Chức năng:

- Helper query string.
- Start HTTP server.

### 6.17. `src/tools/train_rc_car.py`

Trainer baseline supervised dự đoán action trực tiếp.

Hàm:

```python
parse_args()
default_device()
set_seed(seed)
build_model(args)
```

Chức năng:

- Parse config.
- Build `RCDrivingModel`.

Hàm:

```python
action_loss(prediction, target)
```

Chức năng:

- Tính loss action.
- Thường là MSE/L1 tùy implementation.

Hàm:

```python
run_epoch(...)
```

Chức năng:

- Loop train/val supervised.
- Dự đoán action từ ảnh/state.

Hàm:

```python
average_metrics(...)
save_checkpoint(...)
collect_normalization_metadata(...)
args_to_jsonable_dict(...)
epoch_wandb_metrics(...)
main()
```

Chức năng:

- Tương tự trainer khác nhưng cho baseline.

### 6.18. `src/tools/__init__.py`

Package marker cho `tools`. Không chứa logic đáng kể.

## 7. Module `src/viewer`

### 7.1. `src/viewer/index.html`

Vai trò:

- HTML shell cho web viewer.
- Load CSS và JS.
- Chứa layout:
  - danh sách session.
  - panel xem frame/video.
  - control chạy sync/preprocess/extract.
  - vùng progress/log.

### 7.2. `src/viewer/app.js`

Vai trò:

- Frontend logic cho web viewer.

Chức năng chính:

- Fetch danh sách session từ backend.
- Render session list.
- Khi chọn session, fetch frames và metadata.
- Hiển thị frame như video bằng timer/slider.
- Gửi request start job sync/preprocess/extract.
- Poll job status.
- Update progress bar.
- Hiển thị log stdout/stderr.
- Cho chọn preset feature extraction nếu backend expose options.

Điểm cần nhớ:

- Web UI chỉ gọi command local.
- Nếu command fail, cần xem log trong UI hoặc terminal.
- Feature extractor có logic skip cache nếu đã extract đủ và metadata khớp.

### 7.3. `src/viewer/styles.css`

Vai trò:

- Style UI viewer.
- Không ảnh hưởng training.
- Chỉ ảnh hưởng hiển thị.

## 8. Luồng train feature-cache chi tiết

### 8.1. DataLoader tạo batch như thế nào

Từ `create_ac_feature_sequence_dataloaders`:

```text
manifest train/val/test
  -> RCJepaACFeatureSequenceDataset
  -> build_sequence_windows
  -> DataLoader
```

Một sample:

```text
T = raw_frames_per_sample = 8
K = tokens_per_frame = 576
D = embed_dim = 768
latents = [T*K, D] = [4608, 768]
states = [T, 5] = [8, 5]
actions = [T-1, 2] = [7, 2]
```

Một batch train:

```text
latents = [B, 4608, 768]
states = [B, 8, 5]
actions = [B, 7, 2]
```

### 8.2. Teacher forcing đang học gì

Input:

```text
z_0, z_1, ..., z_6
a_0, a_1, ..., a_6
s_0, s_1, ..., s_6
```

Target:

```text
z_1, z_2, ..., z_7
```

Model học:

```text
predictor(z_t, s_t, a_t) -> z_{t+1}
```

### 8.3. Rollout đang học gì

Input thật ban đầu:

```text
z_0
s_0
a_0, a_1, ...
```

Sau đó:

```text
z_1_hat = predictor(z_0, s_0, a_0)
z_2_hat = predictor(z_0, z_1_hat, s_rollout, a_0, a_1)
```

Loss so với:

```text
z_1 thật, z_2 thật, ...
```

Mục tiêu:

- Predictor không chỉ đúng khi thấy latent thật quá khứ.
- Predictor phải chịu lỗi tích lũy khi tự rollout.

### 8.4. Checkpoint và resume

Nếu dừng sau train nhưng trước val:

- `last_train.pt` có `phase = train_complete_waiting_val`.
- Resume từ file này sẽ chạy val của epoch đó, không train lại epoch đó.

Nếu dừng sau epoch hoàn chỉnh:

- `last.pt` có `phase = epoch_complete`.
- Resume từ file này sẽ bắt đầu epoch kế tiếp.

Nếu muốn dùng model tốt nhất:

- `best.pt` là checkpoint có val loss thấp nhất.

### 8.5. W&B

Train log lên W&B:

- Batch metrics:
  - `train_batch/loss`.
  - `train_batch/teacher_forcing_loss`.
  - `train_batch/rollout_loss`.
  - `train_batch/lr`.
  - gradient stats nếu bật.
  - parameter stats nếu bật.

- Epoch metrics:
  - `train/loss`.
  - `val/loss`.
  - `train/teacher_forcing_loss`.
  - `val/teacher_forcing_loss`.
  - `train/rollout_loss`.
  - `val/rollout_loss`.
  - `best/val_loss`.
  - `best/epoch`.
  - `early_stop/patience`.

Để resume cùng W&B run:

- Cần `wandb.run_id` hoặc `--wandb-run-id`.
- Cần `wandb.resume=allow` hoặc CLI tương ứng.
- Cần `wandb.continue_run=true` nếu dùng Hydra config hiện tại.

## 9. Những điểm dễ sai hiện tại

### 9.1. `sequence_stride` không phải frame stride

Hiện tại:

```text
sequence_stride = bước trượt window
```

Không phải:

```text
frame_stride_inside_sample = khoảng cách giữa các frame bên trong sample
```

Nếu muốn sample theo FPS thấp hơn giống source V-JEPA DROID, cần thêm logic mới trong `build_sequence_windows`.

### 9.2. Feature cache hiện đang token-level, không pooled

Mỗi frame lưu:

```text
[576, 768] với ViT-B 384
```

Không phải:

```text
[768]
```

Điều này gần tinh thần V-JEPA AC hơn pooled latent, nhưng nặng hơn rất nhiều.

### 9.3. Eval có thể OOM dù batch nhỏ hơn train

Lý do có thể gồm:

- Attention sequence rất dài.
- PyTorch eval fastpath allocation khác train.
- Fragment CUDA memory.
- W&B watch/grad stats giữ object lâu.
- DataLoader prefetch/pin memory gây pressure.

Biện pháp hiện tại:

- `eval_batch_size` nhỏ, thường `2` cho simple, `1` cho official_lite.
- `maybe_cleanup_cuda()` trước val/test.
- `torch_transformer_eval_fastpath_disabled` cho Simple predictor.

### 9.4. Feature cache fp32 rất lớn

Với mỗi frame ViT-B 384:

```text
576 tokens * 768 dim * 4 bytes ~= 1.77 MB/frame
```

Nếu nhiều chục nghìn frame, cache có thể hàng chục đến hàng trăm GB.

### 9.5. `official_lite` không phải bản Meta y chang

Nó bám các ý chính:

- action/state tokens.
- patch tokens.
- block causal attention mask.
- RoPE.
- token-level prediction.

Nhưng khác:

- Không copy nguyên `VisionTransformerPredictorAC`.
- Không có đầy đủ drop path/stochastic depth như source Meta nếu source dùng.
- Đã điều chỉnh dims/depth nhỏ hơn.
- Dùng feature cache RC, không phải DROID raw video train loop.

### 9.6. Encoder không lưu trong checkpoint predictor

Checkpoint train feature-cache chỉ lưu predictor.

Muốn inference/eval đúng phải dùng cùng:

- feature cache metadata.
- tokens_per_frame.
- embed_dim.
- state/action columns.
- normalizer.

Không được trộn checkpoint predictor train từ ViT-B 384 với feature cache ViT-L/ViT-G hoặc image size khác.

## 10. Lệnh kiểm tra source nhanh

Compile các file chính:

```bash
python3 -m py_compile \
  src/models/rc_jepa_ac.py \
  src/tools/train_rc_jepa_ac_features.py \
  src/tools/train_rc_jepa_ac_features_hydra.py \
  src/tools/extract_vjepa_features.py \
  src/data/feature_sequence_dataset.py \
  src/data/sequence_dataset.py
```

Kiểm tra function/class index:

```bash
rg -n "^(class|def|async def) " src -S
```

Dry-run Hydra tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny \
  runtime.dry_run=true
```

Train tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny
```

Train official-lite tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny
```

## 11. Kết luận kỹ thuật

Source NN-JEPA hiện tại đã có pipeline đầy đủ từ raw data đến feature-cache train/eval/infer/planning. Phần train chính là `train_rc_jepa_ac_features.py`, dataset chính là `feature_sequence_dataset.py`, model chính là `rc_jepa_ac.py`. Cấu hình Hydra giúp thay lệnh dài bằng experiment YAML.

Thiết kế hiện tại đúng hướng cho bài toán RC car world model:

- Dùng frozen V-JEPA 2.1 encoder.
- Dùng token-level latent.
- Conditioning bằng state/action.
- Loss gồm teacher forcing và rollout.
- Có checkpoint/resume/early stopping/W&B.
- Có eval/infer/planning offline.

Những điểm cần theo dõi nếu muốn đưa lên mức chuẩn hơn:

- Thêm frame stride/FPS sampling bên trong sample nếu muốn gần source V-JEPA DROID hơn.
- Cân nhắc session-aware batching nếu I/O/RAM còn bất ổn.
- Cân nhắc extract feature từ raw image kích thước cao hơn thay vì ảnh processed 224 upscale lên 384.
- Nếu dùng official-lite, phải chấp nhận VRAM/RAM cao hơn simple.
- Không trộn feature/checkpoint khác encoder hoặc khác metadata.
