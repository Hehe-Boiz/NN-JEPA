# Báo cáo rà soát data pipeline so với repo JEPA mới

Ngày rà soát: 2026-06-06

## 1. Mục tiêu

Tài liệu này ghi lại kết quả đối chiếu giữa:

- data pipeline của repo hiện tại `NN-JEPA`
- code mới trong repo `JEPA/` vừa được pull về

Mục tiêu là xác định:

- pipeline hiện tại còn phù hợp hay không
- điểm nào đã lệch so với cách thu data mới của `JEPA`
- mức độ ảnh hưởng tới preprocessing, manifest và train
- những phần nào cần ưu tiên sửa hoặc đã được sửa

## 2. Kết luận ngắn

Pipeline hiện tại **không còn hoàn toàn phù hợp nếu giữ nguyên như trước**.

Lý do chính là repo `JEPA` mới đã chuyển sang quy ước:

- session chuẩn sau xử lý dùng `actions_synced.csv`
- dữ liệu IMU chuẩn theo frame dùng `imu_synced.csv`
- `actions.csv` gốc chỉ còn là dữ liệu online/raw để tham chiếu hoặc fallback

Trong khi đó, pipeline cũ của repo này vẫn được viết theo giả định:

- lấy `actions.csv` làm nguồn action chính
- tự ghép `telemetry.csv`, `gyro.csv`, `accel.csv`, `gps.csv` bằng nearest timestamp

Giả định đó không còn là nhánh dữ liệu chuẩn nữa.

## 3. Repo JEPA mới đã thay đổi gì

Sau khi đọc code và tài liệu trong `JEPA/`, các điểm quan trọng là:

### 3.1. Nguồn data chuẩn cho train đã đổi

Repo `JEPA` mới xác định rõ:

- train nên dùng `actions_synced.csv`
- train nên dùng `imu_synced.csv`
- không nên train trực tiếp từ `actions.csv` gốc

Điểm này được nhắc lại nhất quán trong:

- `JEPA/src/sync.py`
- `JEPA/android/README.md`
- `JEPA/HANDOFF.md`
- `JEPA/CLAUDE.md`

### 3.2. Có bước sync offline bắt buộc hơn trước

`JEPA/src/sync.py` không chỉ là tiện ích phụ.

Nó là bước quan trọng để:

- bù `dcam_ms`
- re-pair action theo đúng `t_scene_ms`
- nội suy IMU theo đúng thời điểm cảnh thật
- loại frame ngoài khoảng telemetry hoặc mode không hợp lệ

Nói ngắn gọn:

- dữ liệu raw mới vẫn có giá trị
- nhưng dữ liệu để train phải là dữ liệu **đã sync**

### 3.3. Schema session mới giàu hơn

Session mới của `JEPA` thường có:

- `frames/`
- `actions.csv`
- `telemetry.csv`
- `accel.csv`
- `gyro.csv`
- `rotvec.csv`
- `gps.csv`
- `meta.json`

Sau khi chạy `sync.py` còn có thêm:

- `actions_synced.csv`
- `imu_synced.csv`

Đây là điểm khác biệt lớn nhất so với logic pipeline ban đầu.

## 4. Những chỗ pipeline cũ bị lệch

### 4.1. Chọn sai nguồn action chính

Pipeline cũ chỉ đọc `actions.csv`.

Điều này dẫn tới rủi ro:

- action vẫn là bản online/raw
- chưa chắc khớp đúng thời điểm cảnh thật
- bỏ qua luôn kết quả re-sync chính thức của `JEPA`

Mức độ ảnh hưởng: **cao**

### 4.2. Không hiểu cột `t_scene_ms`

`actions_synced.csv` dùng `t_scene_ms` làm timestamp chính theo milliseconds.

Pipeline cũ chủ yếu quen với:

- `t_ms`
- `timestamp_sec`
- `t_scene`
- `t_pc`

Nếu không hỗ trợ `t_scene_ms`, phần đọc timestamp và match sensor có thể:

- không dùng đúng mốc thời gian
- hoặc fallback sai nhánh

Mức độ ảnh hưởng: **cao**

### 4.3. Không tận dụng `imu_synced.csv`

`sync.py` đã nội suy IMU theo từng frame và ghi ra `imu_synced.csv`.

Nếu pipeline vẫn quay về:

- `gyro.csv`
- `accel.csv`
- nearest match theo timestamp

thì đang bỏ qua bản dữ liệu đã align tốt hơn.

Mức độ ảnh hưởng: **cao**

### 4.4. Có thể để raw telemetry ghi đè dữ liệu synced

Nếu một row đã là dữ liệu sync chuẩn mà code vẫn ưu tiên đọc:

- action từ `telemetry.csv`
- state từ raw sensor stream

thì có thể làm mất lợi ích của bước sync offline.

Mức độ ảnh hưởng: **trung bình đến cao**

### 4.5. Nhánh fallback vẫn chưa nghiêm ngặt như `sync.py`

Trong repo `JEPA`, `sync.py` còn loại:

- frame ngoài khoảng telemetry
- gap telemetry quá lớn
- frame có `mode != 1`

Pipeline fallback từ `actions.csv` trong repo này đơn giản hơn, nên nếu session chưa được sync thì:

- vẫn preprocess được
- nhưng chất lượng dữ liệu không chặt bằng nhánh chuẩn của `JEPA`

Mức độ ảnh hưởng: **trung bình**

## 5. Đánh giá tổng thể

### 5.1. Còn phù hợp ở mức nào

Pipeline hiện tại vẫn còn phù hợp ở 2 việc:

- làm fallback cho session cũ hoặc session chưa chạy `sync.py`
- giữ interface model đơn giản, dễ sửa

### 5.2. Không còn phù hợp ở mức nào

Nó không còn phù hợp nếu mục tiêu là:

- bám đúng data contract mới của repo `JEPA`
- train trên dữ liệu đã được sync chính thức
- tận dụng đúng `dcam_ms`, `actions_synced.csv`, `imu_synced.csv`

Nói gọn:

- **phù hợp cho fallback**
- **không đủ chuẩn nếu xem là pipeline chính thức cho dữ liệu JEPA mới**

## 6. Những phần đã được cập nhật trong repo này

Để bám đúng repo `JEPA` mới, tôi đã cập nhật các điểm sau trong repo `NN-JEPA`:

### 6.1. Ưu tiên file synced

Preprocess giờ ưu tiên:

1. `actions_synced.csv`
2. `actions.csv`

Nếu session đã chạy `JEPA/src/sync.py` thì pipeline sẽ dùng nhánh chuẩn trước.

### 6.2. Hỗ trợ `imu_synced.csv`

Nếu session có `imu_synced.csv`, pipeline sẽ merge dữ liệu IMU theo `frame_idx` vào row action synced trước khi map sang `state`.

### 6.3. Hỗ trợ `t_scene_ms`

Phần đọc timestamp đã hiểu `t_scene_ms`, nên timestamp của session đã sync không còn bị bỏ qua.

### 6.4. Giữ raw làm fallback

Nếu session chưa có file synced:

- pipeline vẫn đọc `actions.csv`
- vẫn match raw telemetry/sensor như trước

Điều này giúp không làm gãy compatibility với dữ liệu cũ.

### 6.5. Thêm test khóa hành vi mới

Đã bổ sung test để đảm bảo:

- `actions_synced.csv` được chọn trước `actions.csv`
- `imu_synced.csv` được merge đúng
- row synced được ưu tiên hơn raw stream
- `t_scene_ms` được đọc đúng

## 7. Rủi ro còn lại

Sau khi vá các điểm chính, vẫn còn một số rủi ro:

### 7.1. `sync.py` chưa được gọi tự động

Hiện tại người dùng vẫn cần chủ động chạy:

```bash
python3 JEPA/src/sync.py
```

trước khi chạy preprocess.

Nếu quên bước này, pipeline sẽ rơi về fallback.

### 7.2. Fallback raw chưa lọc chặt như nhánh chuẩn

Nếu dữ liệu chỉ có `actions.csv`, pipeline vẫn chưa tái hiện đầy đủ toàn bộ logic lọc của `JEPA/src/sync.py`.

### 7.3. `v_t` vẫn là biến yếu

Trong data Android mới, `v_t` thường đi từ `gps.csv.speed`.

Điều này:

- tạm được ngoài trời
- yếu trong nhà

Với nhánh world model AC hiện tại, việc bỏ `v_t` khỏi `AC_STATE_COLUMNS` vẫn là lựa chọn hợp lý.

## 8. Khuyến nghị sử dụng từ bây giờ

Thứ tự đúng nên là:

1. Thu session raw bằng app Android hoặc công cụ tương ứng của `JEPA`
2. Chạy `python3 JEPA/src/sync.py`
3. Chạy `PYTHONPATH=src python3 -m tools.preprocess_data`
4. Train từ manifest đã được tạo ra

Nguyên tắc nên giữ:

- dùng `actions_synced.csv` + `imu_synced.csv` nếu có
- chỉ dùng `actions.csv` như fallback

## 9. Kết luận cuối

Nếu không sửa, data pipeline của repo này đã bắt đầu lệch khỏi hướng phát triển mới của repo `JEPA`.

Sau khi đối chiếu, có thể kết luận:

- repo `JEPA` mới đã nâng chuẩn dữ liệu train lên nhánh **synced offline**
- pipeline của repo này cần bám theo nhánh đó
- các chỗ lệch quan trọng nhất đã được xác định và đã có bản vá tương ứng trong repo hiện tại

Vì vậy, trạng thái hiện tại là:

- **không còn nên dùng `actions.csv` làm nguồn train mặc định**
- **nên coi `actions_synced.csv` + `imu_synced.csv` là chuẩn chính thức**
- **pipeline hiện tại đã được điều chỉnh để theo kịp thay đổi đó**
