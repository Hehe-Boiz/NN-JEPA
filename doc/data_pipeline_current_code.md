# Data Pipeline hiện tại của project

> Update quan trọng: từ ngày **2026-06-05**, repo `JEPA` chuyển hướng chính sang **Android onboard recorder**.
> Nghĩa là session mới không còn chỉ có `frames + actions.csv`, mà thường có thêm
> `telemetry.csv`, `accel.csv`, `gyro.csv`, `rotvec.csv`, `gps.csv`, `meta.json`.
> Code pipeline hiện tại đã được cập nhật để dùng `actions.csv` làm mốc theo frame và ghép các
> stream sensor theo `t_ms` gần nhất.

File này giải thích chi tiết code data pipeline đang nằm ở:

- `src/data/settings.py`
- `src/data/preprocess.py`
- `src/data/dataset.py`
- `src/tools/preprocess_data.py`

Mục tiêu của tài liệu này là:

1. Giải thích code hiện tại đang làm gì, theo đúng implementation đang có.
2. Chỉ rõ những chỗ nào là giả định, những chỗ nào là dữ liệu thật.
3. Chỉ rõ chỗ nào cần sửa khi bạn thay đổi cách thu data hoặc thêm sensor.

---

## 1. Bức tranh tổng thể

Pipeline hiện tại được viết theo hướng đơn giản, dễ sửa tay, không dùng config nhiều tầng.

Luồng xử lý là:

```text
JEPA/data/raw/session_xxx/
  frames/
  actions.csv

-> preprocess_all_sessions()
-> resize ảnh + map CSV về schema model
-> split theo session
-> ghi manifest train/val/test
-> DrivingJEPADataset / DataLoader
```

Schema model mà code đang phục vụ:

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

Điểm rất quan trọng:

- `a_t` hiện tại lấy được khá trực tiếp từ `actions.csv` của recorder.
- `steering_last_t` và `throttle_last_t` có thể suy ra từ action trước đó.
- `v_t`, `yaw_rate_t`, `accel_x_t`, `accel_y_t` hiện chưa được recorder log mặc định.
- Vì vậy nếu session chỉ có `actions.csv`, code sẽ fill các state thiếu bằng giá trị mặc định.

Điều này có nghĩa:

- Pipeline hiện tại chạy được để tổ chức data, train baseline, và giữ interface ổn định.
- Nhưng nếu bạn muốn state vector đúng nghĩa vật lý hơn, bạn cần log thêm sensor thật.

---

## 2. Vì sao pipeline được viết theo kiểu này

Repo `JEPA` hiện tại đã có recorder thực tế ở `JEPA/src/recorder.py`.

Recorder đó lưu data theo kiểu:

```text
JEPA/data/raw/session_YYYYMMDD_HHMMSS/
  frames/
    000001.jpg
    000002.jpg
    ...
  actions.csv
```

Trong `actions.csv` hiện có các cột kiểu:

```csv
frame_idx,t_pc,t_scene,steering,throttle,latency,seq,esp_ms,mode
```

Tức là format raw hiện tại thiên về:

- ảnh
- action điều khiển
- timestamp và metadata đồng bộ

chứ chưa phải full sensor log.

Vì vậy pipeline được viết theo chiến lược sau:

1. Bám đúng data đang có thật.
2. Không ép bạn phải đổi cấu trúc recorder ngay lập tức.
3. Giữ sẵn các cột `s_t` và `a_t` để model phía sau không phải đổi interface nữa.
4. Cho phép sau này chỉ cần log thêm `states.csv` hoặc `signals.csv` là pipeline đọc tiếp được.

---

## 3. Cấu trúc file và vai trò từng file

### `src/data/settings.py`

Đây là file quan trọng nhất nếu bạn muốn chỉnh pipeline.

Nó đóng vai trò:

- chứa toàn bộ biến toàn cục
- là nơi đổi đường dẫn
- là nơi đổi cách đọc CSV
- là nơi đổi resize, split, filtering, augmentation

Bạn có thể xem nó như “control panel” của pipeline.

### `src/data/preprocess.py`

Đây là phần xử lý offline.

Nó làm các việc:

- tìm session raw
- tìm file CSV đúng trong session
- tìm ảnh trong `frames/`
- đọc CSV
- map dữ liệu CSV sang state/action theo schema model
- resize ảnh
- loại bớt dữ liệu lỗi hoặc không hợp lệ
- split train/val/test theo session
- ghi manifest ra đĩa
- ghi report thống kê

### `src/data/dataset.py`

Đây là phần dùng lúc train.

Nó làm các việc:

- đọc manifest
- load ảnh đã preprocess
- augment ảnh ở split train
- normalize ảnh
- convert state/action thành tensor PyTorch
- tạo `DataLoader`

### `src/tools/preprocess_data.py`

Đây là CLI nhỏ để gọi preprocessing bằng command line:

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

Nếu chưa có data raw thì nó báo lỗi gọn bằng JSON.

---

## 4. Giải thích chi tiết `settings.py`

`settings.py` là nơi bạn nên mở đầu tiên mỗi khi muốn đổi pipeline.

### 4.1. Nhóm đường dẫn

```python
REPO_ROOT = Path(__file__).resolve().parents[3]
JEPA_ROOT = REPO_ROOT / "JEPA"

RAW_DATA_DIR = JEPA_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = JEPA_ROOT / "data" / "processed"
PROCESSED_IMAGE_DIR = PROCESSED_DATA_DIR / "images"
MANIFEST_DIR = PROCESSED_DATA_DIR / "manifests"
REPORT_DIR = PROCESSED_DATA_DIR / "reports"
```

Ý nghĩa:

- `RAW_DATA_DIR`: nơi recorder đang đổ session thô.
- `PROCESSED_DATA_DIR`: nơi pipeline ghi output sau xử lý.
- `PROCESSED_IMAGE_DIR`: ảnh đã resize.
- `MANIFEST_DIR`: file `train.jsonl`, `val.jsonl`, `test.jsonl`.
- `REPORT_DIR`: log thống kê preprocess.

Nếu sau này bạn muốn đổi toàn bộ data sang ổ cứng khác hoặc folder khác, chỉnh ở đây là đủ.

### 4.2. Nhóm cấu trúc session

```python
SESSION_GLOB = "session_*"
FRAME_DIR_NAME = "frames"
FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png")
CSV_CANDIDATES = ("states.csv", "signals.csv", "actions.csv")
```

Ý nghĩa:

- chỉ đọc những thư mục tên kiểu `session_*`
- mặc định ảnh nằm trong `frames/`
- hỗ trợ ảnh `jpg/jpeg/png`
- khi tìm CSV thì ưu tiên:
  1. `states.csv`
  2. `signals.csv`
  3. `actions.csv`

Lý do của `CSV_CANDIDATES`:

- Hiện tại repo chỉ có `actions.csv`.
- Sau này nếu bạn log đầy đủ hơn và tạo `states.csv` hoặc `signals.csv`, pipeline sẽ tự ưu tiên file đầy đủ hơn.

### 4.3. Nhóm định nghĩa schema model

```python
STATE_COLUMNS = (
    "v_t",
    "yaw_rate_t",
    "accel_x_t",
    "accel_y_t",
    "steering_last_t",
    "throttle_last_t",
)

ACTION_COLUMNS = (
    "steering_cmd_t",
    "throttle_cmd_t",
)
```

Đây là định nghĩa chuẩn của input vector model.

Dataset luôn xuất tensor theo đúng thứ tự này.

Điều này rất quan trọng vì:

- model train theo thứ tự cố định
- nếu sau này đổi thứ tự mà không kiểm soát, model sẽ học sai

### 4.4. Nhóm map cột CSV sang schema model

```python
FRAME_INDEX_KEYS = ("frame_idx",)
TIMESTAMP_KEYS = ("timestamp_sec", "t_scene", "t_pc")
```

`FRAME_INDEX_KEYS` cho biết cột nào dùng để nối row CSV với frame.

`TIMESTAMP_KEYS` cho biết cột nào có thể dùng làm timestamp của sample. Code ưu tiên theo thứ tự:

1. `timestamp_sec`
2. `t_scene`
3. `t_pc`

Tiếp theo là action mapping:

```python
ACTION_SOURCE_KEYS = {
    "steering_cmd_t": ("steering_cmd_t", "steering"),
    "throttle_cmd_t": ("throttle_cmd_t", "throttle"),
}
```

Ý nghĩa:

- nếu CSV đã có cột đúng tên model thì dùng luôn
- nếu chưa có thì fallback sang tên cột thực tế hiện tại của recorder

State mapping:

```python
STATE_SOURCE_KEYS = {
    "v_t": ("v_t", "speed", "velocity"),
    "yaw_rate_t": ("yaw_rate_t", "yaw_rate", "gyro_z"),
    "accel_x_t": ("accel_x_t", "accel_x", "ax"),
    "accel_y_t": ("accel_y_t", "accel_y", "ay"),
    "steering_last_t": ("steering_last_t", "steering_last"),
    "throttle_last_t": ("throttle_last_t", "throttle_last"),
}
```

Ý nghĩa:

- code chấp nhận nhiều alias tên cột
- giúp bạn không phải sửa logic ngay nếu logger đặt tên hơi khác

Ví dụ:

- bạn log `ax` thay vì `accel_x_t` thì pipeline vẫn đọc được
- bạn log `gyro_z` thay vì `yaw_rate_t` thì pipeline vẫn đọc được

### 4.5. Nhóm chính sách khi thiếu state

```python
ALLOW_ACTIONS_ONLY_SESSIONS = True
MISSING_STATE_VALUE = 0.0
USE_PREVIOUS_ACTION_AS_LAST_CONTROL = True
DEFAULT_STEERING_LAST = 0.0
DEFAULT_THROTTLE_LAST = 0.0
```

Đây là nhóm rất quan trọng.

Ý nghĩa:

- `ALLOW_ACTIONS_ONLY_SESSIONS = True`
  Cho phép session chỉ có action mà không có state sensor thật.

- `MISSING_STATE_VALUE = 0.0`
  Nếu thiếu `v_t`, `yaw_rate_t`, `accel_x_t`, `accel_y_t`, code sẽ điền `0.0`.

- `USE_PREVIOUS_ACTION_AS_LAST_CONTROL = True`
  Nếu thiếu `steering_last_t` hoặc `throttle_last_t`, code lấy action trước đó làm “last control”.

- `DEFAULT_STEERING_LAST` và `DEFAULT_THROTTLE_LAST`
  Dùng cho sample đầu tiên hoặc khi không muốn lấy previous action.

Ý nghĩa thực tế:

- state hiện tại chưa đầy đủ vật lý
- nhưng pipeline vẫn giữ đúng số chiều đầu vào cho model
- sau này thay data thật vào sẽ ít phải đổi code

### 4.6. Nhóm cleaning / filtering

```python
USE_EVERY_NTH_FRAME = 1
MIN_SESSION_SAMPLES = 8
DROP_DUPLICATE_FRAME_INDEX = True
DROP_ROWS_WITH_MISSING_FRAME = True
DROP_ROWS_WITH_MISSING_ACTION = True
DROP_ROWS_OUTSIDE_ACTION_RANGE = True
REMOVE_SIMPLE_OUTLIERS = False
OUTLIER_STD_FACTOR = 4.0
OUTLIER_COLUMNS = ("v_t", "yaw_rate_t", "accel_x_t", "accel_y_t")
```

Giải thích:

- `USE_EVERY_NTH_FRAME`
  Nếu bằng `1` thì lấy mọi row.
  Nếu bằng `2` thì lấy 1 row, bỏ 1 row.
  Dùng khi bạn muốn downsample dataset nhanh.

- `MIN_SESSION_SAMPLES`
  Session ít hơn ngưỡng này sẽ bị bỏ hẳn.

- `DROP_DUPLICATE_FRAME_INDEX`
  Nếu cùng `frame_idx` xuất hiện nhiều lần thì giữ sample đầu, bỏ các sample sau.

- `DROP_ROWS_WITH_MISSING_FRAME`
  Nếu row CSV trỏ tới frame không tồn tại thì bỏ.

- `DROP_ROWS_WITH_MISSING_ACTION`
  Nếu thiếu steering/throttle thì bỏ row.

- `DROP_ROWS_OUTSIDE_ACTION_RANGE`
  Nếu action nằm ngoài khoảng hợp lệ thì bỏ.

- `REMOVE_SIMPLE_OUTLIERS`
  Nếu bật lên, code sẽ loại outlier thô bằng mean ± `OUTLIER_STD_FACTOR * std`.

Nhóm range action:

```python
STEERING_MIN = -1.0
STEERING_MAX = 1.0
THROTTLE_MIN = -1.0
THROTTLE_MAX = 1.0
```

Vì firmware và recorder hiện dùng steering/throttle normalized trong khoảng `[-1, 1]`, đây là range hợp lý.

### 4.7. Nhóm scale action

```python
STEERING_SCALE = 1.0
THROTTLE_SCALE = 1.0
```

Lý do tồn tại:

Trong `PLAN.md` có ghi chú rằng throttle thực tế có thể có biên độ nhỏ hơn steering.

Hai hệ số này cho phép bạn:

- giữ raw log nguyên gốc
- nhưng scale action khi đưa vào pipeline
- thuận tiện cho ablation mà không cần rewrite data

Ví dụ:

- nếu thấy throttle quá nhỏ, có thể thử `THROTTLE_SCALE = 2.0`
- nếu steering quá nhạy, có thể thử `STEERING_SCALE = 0.8`

### 4.8. Nhóm ảnh

```python
RESIZE_IMAGES = True
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_FORMAT = "jpg"
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
```

Ý nghĩa:

- preprocess resize ảnh offline về `224x224`
- lưu lại dạng `jpg`
- normalize theo mean/std kiểu ImageNet trong dataset

Vì sao resize offline:

- giảm chi phí train
- manifest ổn định
- ảnh train/val/test thống nhất kích thước

Vì sao normalize online:

- giữ code dataset rõ ràng
- dễ sửa mean/std nếu cần

### 4.9. Nhóm split

```python
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42
```

Code split theo session, không split theo frame.

Điều này đúng với bài toán điều khiển hơn vì:

- cùng một quỹ đạo có các frame rất giống nhau
- nếu chia theo frame thì train và val dễ bị leakage

### 4.10. Nhóm DataLoader

```python
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
SHUFFLE_TRAIN = True
```

Đây là tham số runtime cho train.

### 4.11. Nhóm augmentation

```python
BRIGHTNESS_JITTER = 0.15
CONTRAST_JITTER = 0.15
SATURATION_JITTER = 0.05
GAUSSIAN_BLUR_PROB = 0.1
GAUSSIAN_BLUR_RADIUS = 1.0
HORIZONTAL_FLIP_PROB = 0.0
```

`HORIZONTAL_FLIP_PROB` mặc định để `0.0`.

Lý do:

- với điều khiển xe, lật ngang ảnh không đơn giản như classification
- nếu lật, phải đổi dấu đúng các biến liên quan đến hướng
- nếu hệ trục sensor hoặc steering sign không đúng giả định, flip có thể phá nhãn

### 4.12. Hàm `make_output_dirs()`

Hàm này chỉ có một nhiệm vụ:

- đảm bảo tất cả thư mục output tồn tại trước khi preprocess chạy

---

## 5. Giải thích chi tiết `preprocess.py`

Đây là file quan trọng nhất về logic pipeline.

### 5.1. `preprocess_all_sessions()`

Đây là entrypoint chính của preprocessing.

Nó làm theo thứ tự:

1. Tạo thư mục output bằng `settings.make_output_dirs()`
2. Tìm tất cả session raw qua `find_session_dirs()`
3. Với từng session, gọi `preprocess_one_session()`
4. Bỏ các session không đủ sample
5. Split session thành train/val/test bằng `build_session_split()`
6. Ghi manifest `.jsonl` cho từng split
7. Tính thống kê feature bằng `compute_feature_stats()`
8. Ghi report JSON ra `reports/preprocess_report.json`

Output trả về là một dict `summary`.

Dict này có các phần:

- `raw_data_dir`
- `processed_data_dir`
- `counts`
- `sessions`
- `feature_stats`
- `session_reports`

### 5.2. `find_session_dirs()`

Hàm này:

- duyệt `settings.RAW_DATA_DIR`
- lấy tất cả thư mục match `settings.SESSION_GLOB`

Hiện mặc định là `session_*`.

### 5.3. `preprocess_one_session(session_dir)`

Đây là phần xử lý core của từng session.

Trình tự xử lý:

1. Tìm CSV bằng `find_csv_file()`
2. Tạo map `frame_index -> frame_path` bằng `build_frame_map()`
3. Đọc toàn bộ row CSV bằng `read_csv_rows()`
4. Duyệt từng row
5. Áp dụng stride nếu có
6. Lấy `frame_index`
7. Bỏ row nếu duplicate frame index
8. Bỏ row nếu frame không tồn tại
9. Đọc action
10. Scale action
11. Kiểm tra range action
12. Đọc hoặc suy state
13. Resize và ghi ảnh processed
14. Tạo sample dict
15. Sau cùng nếu bật thì remove outlier
16. Nếu session còn ít hơn `MIN_SESSION_SAMPLES` thì bỏ luôn session đó

Hàm này cũng tạo `report` riêng cho session.

`report` theo dõi:

- tổng số row raw
- số row bị bỏ vì thiếu frame
- số row bị bỏ vì thiếu action
- số row bị bỏ vì duplicate frame
- số row bị bỏ vì action ngoài range
- số row bị bỏ vì stride
- số row phải fill state thiếu
- cột state nào bị thiếu

Đây là phần rất hữu ích khi bạn bắt đầu thu data thật.

### 5.4. `find_csv_file(session_dir)`

Logic rất đơn giản:

- lần lượt check `states.csv`
- nếu không có thì check `signals.csv`
- nếu không có thì check `actions.csv`
- nếu không có cái nào thì raise lỗi

Mục đích:

- dùng được ngay với recorder hiện tại (`actions.csv`)
- nhưng vẫn sẵn sàng cho logger đầy đủ hơn sau này

### 5.5. `build_frame_map(frames_dir)`

Mục tiêu của hàm này là nối được row CSV với file ảnh thật.

Nó:

1. đọc mọi file trong `frames/`
2. sort theo `natural_sort_key()`
3. lấy số trong tên file nếu có
4. map `frame_index -> path`

Ví dụ:

- `000001.jpg` -> `1`
- `000245.jpg` -> `245`

Nếu tên file không có số, code fallback sang thứ tự duyệt file.

### 5.6. `read_csv_rows(csv_path)`

Chỉ đọc CSV bằng `csv.DictReader` rồi trả về list of dict.

Không dùng pandas để code nhẹ và dễ đọc hơn.

### 5.7. `get_frame_index(row, row_number)`

Ý nghĩa:

- ưu tiên đọc `frame_idx` từ CSV
- nếu đọc được thì convert sang `int`
- nếu không có thì fallback sang `row_number + 1`

Fallback này giúp code vẫn chạy khi CSV không có frame index rõ ràng, miễn frame và row đi cùng thứ tự.

### 5.8. `read_timestamp(row)`

Hàm này lần lượt thử các cột trong:

```python
TIMESTAMP_KEYS = ("timestamp_sec", "t_scene", "t_pc")
```

Nó trả về giá trị timestamp đầu tiên đọc được.

Trong bối cảnh recorder hiện tại:

- `t_scene` thường có ý nghĩa tốt hơn cho đồng bộ ảnh-action
- `t_pc` là fallback

### 5.9. `read_action(row)`

Hàm này map row CSV sang action model.

Ví dụ với CSV hiện tại:

```csv
steering,throttle
0.12,-0.08
```

nó sẽ tạo:

```python
{
  "steering_cmd_t": 0.12,
  "throttle_cmd_t": -0.08,
}
```

Nếu thiếu action và `DROP_ROWS_WITH_MISSING_ACTION = True` thì row bị bỏ.

### 5.10. `read_state(row, previous_action)`

Đây là chỗ quan trọng nhất về mặt ý nghĩa dữ liệu.

Nó tạo state vector `s_t`.

#### 5.10.1. Phần state sensor

Cho 4 biến:

- `v_t`
- `yaw_rate_t`
- `accel_x_t`
- `accel_y_t`

code sẽ thử đọc từ CSV bằng alias tương ứng.

Nếu không tìm thấy:

- nếu `ALLOW_ACTIONS_ONLY_SESSIONS = False` thì raise lỗi
- nếu `ALLOW_ACTIONS_ONLY_SESSIONS = True` thì fill `MISSING_STATE_VALUE`

#### 5.10.2. Phần last control

Cho 2 biến:

- `steering_last_t`
- `throttle_last_t`

code thử đọc trực tiếp từ CSV.

Nếu không có:

- nếu `USE_PREVIOUS_ACTION_AS_LAST_CONTROL = True`
  thì lấy từ `previous_action`
- nếu không
  thì dùng `DEFAULT_STEERING_LAST` và `DEFAULT_THROTTLE_LAST`

Lý do logic này khá hợp lý:

- trong dữ liệu điều khiển, action ngay trước đó thường là xấp xỉ tốt cho “last control”
- đặc biệt khi logger chưa ghi riêng `steering_last_t` và `throttle_last_t`

### 5.11. `keep_meta_fields(row)`

Hàm này không tác động vào tensor train trực tiếp, nhưng giữ lại metadata hữu ích.

Hiện nó lưu nếu có:

- `t_pc`
- `t_scene`
- `latency`
- `seq`
- `esp_ms`
- `mode`

Điều này giúp sau này bạn:

- debug session
- trace lại latency
- kiểm tra mode record
- đối chiếu với firmware log

### 5.12. `action_in_valid_range(action)`

Hàm này chỉ check xem:

- steering có nằm trong `[STEERING_MIN, STEERING_MAX]`
- throttle có nằm trong `[THROTTLE_MIN, THROTTLE_MAX]`

Nếu không hợp lệ thì bỏ row.

### 5.13. `prepare_image(source_path, output_path)`

Hàm này:

1. mở ảnh bằng PIL
2. convert sang RGB
3. resize nếu `RESIZE_IMAGES = True`
4. save ra folder processed

Pipeline hiện không làm augmentation ở đây.

Augmentation được để sang lúc train.

### 5.14. `remove_simple_outliers(samples)`

Đây là bộ lọc outlier đơn giản kiểu:

```text
keep if mean - k*std <= value <= mean + k*std
```

với `k = OUTLIER_STD_FACTOR`.

Nó chỉ áp dụng cho các cột trong `OUTLIER_COLUMNS`.

Mặc định đang tắt:

```python
REMOVE_SIMPLE_OUTLIERS = False
```

Lý do hợp lý:

- dataset thực tế chưa chắc đủ lớn
- nếu state đang bị fill 0 nhiều thì outlier filter kiểu này chưa chắc hữu ích

### 5.15. `build_session_split(session_ids)`

Hàm này split theo session.

Nó xử lý cả case dataset rất nhỏ:

- 1 session -> train
- 2 session -> train + test
- >=3 session -> train/val/test theo tỷ lệ cấu hình

Mục đích:

- tránh leakage theo quỹ đạo
- vẫn chạy được ngay cả khi dataset ít

### 5.16. `compute_feature_stats(session_samples)`

Hàm này tính:

- mean
- std
- min
- max

cho tất cả cột state/action đang có.

Thống kê này được lưu vào report để:

- kiểm tra action có bị lệch không
- kiểm tra state fill 0 có chiếm quá nhiều không
- hỗ trợ normalize hoặc rescale sau này

### 5.17. `natural_sort_key(path)`

Hàm sort tên file theo kiểu con người mong muốn.

Ví dụ:

- `2.jpg` đứng trước `10.jpg`

thay vì sort chuỗi thuần.

### 5.18. `compute_std(values)`

Hàm tự tính standard deviation đơn giản, không cần numpy/pandas.

### 5.19. `extract_digits(text)`

Lấy toàn bộ chữ số từ tên file.

Ví dụ:

- `"000123"` -> `123`
- `"frame_45"` -> `45`

### 5.20. `read_first_float(row, keys)`

Đây là helper rất quan trọng.

Nó lần lượt thử nhiều tên cột cho cùng một giá trị logic.

Ví dụ:

- với `yaw_rate_t`, nó có thể thử đọc từ `yaw_rate_t`, `yaw_rate`, hoặc `gyro_z`

Nhờ đó pipeline mềm hơn trước nhiều kiểu logger khác nhau.

---

## 6. Giải thích chi tiết `dataset.py`

`dataset.py` là phần dùng khi train bằng PyTorch.

### 6.1. Import torch trực tiếp

Hiện tại code import `torch` trực tiếp ở đầu file.

Điều này có nghĩa:

- dataset được xem là thành phần chắc chắn dành cho training/inference PyTorch
- nếu môi trường thiếu `torch`, module sẽ fail ngay lúc import

Đây là lựa chọn hợp lý nếu bạn xác định chắc chắn code này chỉ chạy trong môi trường ML đã cài PyTorch.

### 6.2. `HORIZONTAL_FLIP_SIGN_COLUMNS`

```python
HORIZONTAL_FLIP_SIGN_COLUMNS = {
    "yaw_rate_t": -1.0,
    "accel_y_t": -1.0,
    "steering_last_t": -1.0,
    "steering_cmd_t": -1.0,
}
```

Nếu ảnh bị flip ngang, các biến liên quan đến hướng trái/phải phải đổi dấu.

Code hiện đang giả định:

- `yaw_rate_t` đổi dấu khi flip
- `accel_y_t` đổi dấu khi flip
- `steering_last_t` đổi dấu khi flip
- `steering_cmd_t` đổi dấu khi flip

Đây là giả định hợp lý về mặt vật lý, nhưng bạn chỉ nên bật flip khi chắc chắn hệ trục của mình đúng như vậy.

### 6.3. `TrainAugmentor`

Class này chỉ dùng cho split `train`.

Nó áp dụng augmentation online:

- horizontal flip
- brightness jitter
- contrast jitter
- saturation jitter
- gaussian blur

Quan trọng:

- không có random crop
- không có rotate
- không có heavy augmentation

Lý do:

- bài toán điều khiển nhạy với hình học
- augment quá mạnh có thể phá ý nghĩa action

### 6.4. `DrivingJEPADataset`

Đây là class dataset chính.

#### `__init__(split, manifest_path=None)`

Nó:

- nhận tên split `train` / `val` / `test`
- mặc định tự tìm manifest ở `settings.MANIFEST_DIR`
- load toàn bộ sample vào `self.samples`
- nếu là `train` thì bật augmentor

#### `__len__()`

Trả về số sample trong manifest.

#### `__getitem__(index)`

Đây là hàm quan trọng nhất.

Cho mỗi sample, nó làm:

1. mở ảnh từ `frame_path`
2. convert sang RGB
3. copy state/action dict
4. nếu là train thì augment ảnh và update state/action khi cần
5. convert ảnh sang tensor bằng `image_to_tensor()`
6. normalize ảnh bằng `normalize_tensor()`
7. convert state thành tensor theo thứ tự `settings.STATE_COLUMNS`
8. convert action thành tensor theo thứ tự `settings.ACTION_COLUMNS`
9. trả về dict kết quả

Dict trả về có dạng:

```python
{
  "image": image_tensor,
  "state": state_tensor,
  "action": action_tensor,
  "sample_id": ...,
  "session_id": ...,
  "frame_index": ...,
  "timestamp_sec": ...,
}
```

Điều này có nghĩa:

- phần model có thể lấy `batch["image"]`, `batch["state"]`, `batch["action"]`
- phần debug vẫn có `sample_id`, `session_id`, `frame_index`

### 6.5. `load_manifest(path)`

Manifest được lưu theo định dạng JSON Lines (`.jsonl`).

Mỗi dòng là 1 sample JSON độc lập.

Ưu điểm:

- dễ inspect bằng text editor
- dễ append
- dễ debug

### 6.6. `image_to_tensor(image)`

Hàm này:

1. convert PIL image sang numpy array float32
2. scale pixel từ `[0, 255]` về `[0, 1]`
3. đổi shape từ `HWC` sang `CHW`
4. tạo `torch.Tensor`

### 6.7. `normalize_tensor(tensor, mean, std)`

Hàm này làm normalize chuẩn:

```text
(tensor - mean) / std
```

theo từng channel RGB.

### 6.8. `create_dataloaders(batch_size=None, num_workers=None)`

Hàm này tạo luôn 3 `DataLoader`:

- `train`
- `val`
- `test`

Nó dùng default từ `settings.py` nếu bạn không truyền tay.

Đây là interface tiện vì code train có thể làm:

```python
dataloaders = create_dataloaders()
train_loader = dataloaders["train"]
```

### 6.9. `_require_numpy()`

`numpy` vẫn được import lười ở đây.

Lý do:

- numpy chỉ thực sự cần lúc convert ảnh sang tensor
- nếu thiếu numpy thì báo lỗi đúng chỗ đang dùng

---

## 7. Giải thích `preprocess_data.py`

File này rất nhỏ.

Nó chỉ:

1. gọi `preprocess_all_sessions()`
2. in summary dạng JSON
3. nếu lỗi vì chưa có data hoặc không có sample hợp lệ, in error JSON và thoát với code `1`

Mục tiêu:

- CLI ngắn
- dễ gọi từ terminal
- log rõ ràng, dễ đọc và dễ parse

Command dùng:

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

---

## 8. Giải thích `__init__.py`

File `src/data/__init__.py` chỉ export các thành phần chính:

- `ACTION_COLUMNS`
- `STATE_COLUMNS`
- `DrivingJEPADataset`
- `create_dataloaders`
- `preprocess_all_sessions`

Mục đích:

- giúp import gọn hơn
- phần code train không cần biết sâu cấu trúc thư mục con

Ví dụ:

```python
from data import create_dataloaders, preprocess_all_sessions
```

---

## 9. Manifest hiện tại trông như thế nào

Sau preprocess, mỗi sample trong manifest có dạng gần như sau:

```json
{
  "sample_id": "session_20260604_120000_000123",
  "session_id": "session_20260604_120000",
  "frame_index": 123,
  "timestamp_sec": 1717470001.25,
  "frame_path": ".../JEPA/data/processed/images/session_20260604_120000/000123.jpg",
  "source_frame_path": ".../JEPA/data/raw/session_20260604_120000/frames/000123.jpg",
  "state": {
    "v_t": 0.0,
    "yaw_rate_t": 0.0,
    "accel_x_t": 0.0,
    "accel_y_t": 0.0,
    "steering_last_t": 0.11,
    "throttle_last_t": 0.18
  },
  "action": {
    "steering_cmd_t": 0.12,
    "throttle_cmd_t": 0.20
  },
  "meta": {
    "t_pc": 1717470001.36,
    "t_scene": 1717470001.25,
    "latency": 0.11,
    "seq": 124,
    "esp_ms": 555123,
    "mode": 1
  },
  "split": "train"
}
```

Ý nghĩa:

- sample này vừa đủ cho phần model dùng ngay
- metadata vẫn được giữ để debug và phân tích sau này

---

## 10. Những giả định quan trọng của code hiện tại

### 10.1. Giả định về data raw

Code giả định session nằm trong:

```text
JEPA/data/raw/session_xxx/
```

và có:

- `frames/`
- ít nhất một trong `states.csv`, `signals.csv`, `actions.csv`

### 10.2. Giả định về frame index

Code giả định:

- `frame_idx` trong CSV tương ứng với tên ảnh
- hoặc ít nhất thứ tự row tương ứng với thứ tự frame

Nếu recorder sau này đổi cách đánh số frame, bạn phải xem lại `get_frame_index()` và `build_frame_map()`.

### 10.3. Giả định về action range

Code giả định steering/throttle normalized trong `[-1, 1]`.

Nếu logger của bạn đổi sang:

- byte `0..255`
- PWM `1000..2000`
- hoặc microsecond raw

thì bạn phải đổi lại mapping trong `read_action()` hoặc chỉnh preprocessing trước đó.

### 10.4. Giả định về state thiếu

Code hiện chấp nhận thiếu state sensor thật.

Điều này giúp pipeline chạy, nhưng có nhược điểm:

- model có thể học chủ yếu từ ảnh + action
- các chiều state bị fill 0 không mang nhiều thông tin thực

Nói ngắn gọn:

- interface đúng
- semantics chưa đủ mạnh nếu không log sensor thật

---

## 11. Những hạn chế hiện tại

### Hạn chế 1: chưa có state thật cho 4 biến đầu

Đây là hạn chế lớn nhất.

Nếu bạn thật sự muốn dùng:

- `v_t`
- `yaw_rate_t`
- `accel_x_t`
- `accel_y_t`

thì recorder hoặc firmware phải log thêm.

### Hạn chế 2: chưa có temporal window

Dataset hiện tại là sample đơn:

```text
image_t, state_t, action_t
```

chứ chưa phải sequence:

```text
image_{t-k:t}, state_{t-k:t}, action_{t-k:t}
```

Điều này đủ cho baseline đơn giản, nhưng chưa phải dạng temporal world-model đầy đủ.

### Hạn chế 3: outlier filter còn rất đơn giản

Hiện chỉ có mean/std thô, chưa có:

- robust z-score
- percentile clip
- session-aware cleaning
- logic phát hiện đứng yên quá lâu

### Hạn chế 4: chưa có balancing theo mode hoặc route

Ví dụ:

- rẽ trái nhiều hơn rẽ phải
- đứng yên nhiều hơn chạy
- indoor room A nhiều hơn room B

Pipeline hiện chưa cân bằng phân bố này.

---

## 12. Khi nào bạn nên sửa file nào

### Sửa `settings.py` khi:

- đổi đường dẫn data
- đổi size ảnh
- đổi split ratio
- đổi batch size
- đổi stride lấy mẫu
- đổi policy fill state thiếu
- đổi augmentation

### Sửa `preprocess.py` khi:

- đổi format session
- đổi format CSV
- thêm nhiều file metadata khác
- muốn sequence sampling phức tạp hơn
- muốn logic cleaning mạnh hơn

### Sửa `dataset.py` khi:

- muốn trả về thêm trường
- muốn dùng sequence thay vì single frame
- muốn normalize khác
- muốn augment khác

### Sửa `JEPA/src/recorder.py` hoặc firmware khi:

- muốn state thật thay vì fill 0
- muốn log IMU / encoder / odometry
- muốn thêm quality flags
- muốn thêm domain labels

---

## 13. Cách nâng cấp pipeline theo hướng đúng nhất cho đề tài

Thứ tự nâng cấp hợp lý:

### Bước 1. Log state thật

Ưu tiên thêm vào logger:

- `v_t`
- `yaw_rate_t`
- `accel_x_t`
- `accel_y_t`

Nếu có thể, lưu vào `states.csv` hoặc `signals.csv`.

### Bước 2. Giữ `steering_last_t`, `throttle_last_t` là cột thật

Hiện pipeline đang suy từ previous action. Cách này ổn cho baseline, nhưng log trực tiếp vẫn chuẩn hơn.

### Bước 3. Thêm sequence dataset

Cho JEPA/world model, rất có thể bạn sẽ cần:

- frame `t`
- frame `t+1`
- hoặc window `t-k ... t`

Khi đó dataset nên chuyển từ sample đơn sang temporal sample.

### Bước 4. Thêm split rule tốt hơn

Ví dụ:

- split theo nhà
- split theo ngày thu data
- split theo route

để đánh giá generalization nghiêm túc hơn.

---

## 14. Test hiện tại đang kiểm tra gì

Trong `tests/test_pipeline_simple.py`, hiện có 4 nhóm test:

1. schema state/action đúng thứ tự mong muốn
2. split theo session vẫn hợp lệ khi dataset nhỏ
3. row của `actions.csv` map đúng sang action/state
4. horizontal flip đổi dấu đúng các biến liên quan đến hướng

Điều test này đảm bảo:

- code không bị lệch interface model
- mapping từ recorder hiện tại sang schema model vẫn đúng

Nó chưa test:

- preprocess end-to-end với session thật
- training runtime với torch thật
- performance / throughput

---

## 15. Cách đọc nhanh code nếu bạn quay lại sau này

Nếu vài tuần nữa bạn quay lại và muốn nhớ nhanh:

1. Mở `settings.py` để xem pipeline đang được cấu hình thế nào.
2. Mở `preprocess.py`, đọc `preprocess_all_sessions()` trước.
3. Sau đó đọc `preprocess_one_session()` vì đây là lõi logic.
4. Cuối cùng đọc `dataset.py` để xem train lấy batch ra sao.

Nếu muốn sửa ít:

- thường chỉ cần đụng `settings.py`

Nếu muốn đổi workflow dữ liệu:

- gần như chắc chắn phải sửa `preprocess_one_session()`

---

## 16. Kết luận ngắn

Code hiện tại được tối ưu cho:

- dễ hiểu
- dễ sửa tay
- bám đúng data recorder hiện có
- giữ interface ổn định cho model phía sau

Nó chưa phải pipeline “hoàn hảo” cho full-state world model, vì recorder hiện tại chưa log đủ state sensor thật.

Nhưng nó là một nền hợp lý để:

- thu data ngay
- train baseline ngay
- sau đó nâng cấp dần logger và state vector mà không phải đập bỏ toàn bộ pipeline.
