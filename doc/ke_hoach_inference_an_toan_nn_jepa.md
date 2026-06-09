# Kế hoạch inference an toàn cho NN-JEPA

## Bối cảnh hiện tại

NN-JEPA hiện đang train predictor/world model từ feature cache đã trích xuất bằng encoder V-JEPA 2.1 frozen. Pipeline hiện tại có hai phần quan trọng:

- Encoder V-JEPA 2.1 đã chạy riêng để tạo feature cache.
- Predictor JEPA-AC của NN-JEPA nhận latent tokens + state + action, rồi dự đoán latent tokens tương lai.

Điểm quan trọng là model hiện tại chưa phải policy trực tiếp kiểu `image -> steering/throttle`. Nó là world model: nếu đưa vào trạng thái hiện tại và một chuỗi action giả định, nó dự đoán latent tương lai sẽ ra sao. Vì vậy inference đúng hướng không phải chỉ `forward` một lần để lấy action, mà cần thêm một planner tìm action làm latent dự đoán tiến gần tới goal.

## Vì sao không nên nhảy thẳng vào closed-loop

Closed-loop nghĩa là model đang chạy thật trên xe, nhận camera/sensor realtime, sinh action, rồi gửi ngay xuống servo/throttle. Đây là bước rủi ro cao vì bất kỳ lỗi nào trong preprocess, scale action, state normalization, latency, hoặc truyền lệnh đều có thể làm xe chạy sai.

Các lỗi dễ gặp nếu nhảy thẳng vào closed-loop:

- Feature format sai: dùng checkpoint train với `576 token/frame, D=768` nhưng inference lại đưa `256 token/frame, D=1024`.
- State/action normalization sai: model train bằng action/state đã chuẩn hóa nhưng inference gửi action raw trực tiếp vào predictor.
- Goal latent không rõ: planner không biết cần đi tới đâu nếu không có goal frame/subgoal latent.
- Latency camera/điều khiển cao: action sinh ra dựa trên frame cũ, xe đã ở trạng thái khác.
- Không có watchdog: nếu process inference treo, xe có thể giữ throttle/steering cũ.
- Không có neutral fallback: khi mất frame, mất sensor, OOM, NaN, hoặc planner fail thì xe không tự về trạng thái an toàn.
- Không giới hạn throttle: model có thể chọn action hợp lệ theo toán học nhưng quá mạnh với xe thật.

Vì vậy lộ trình đúng là đi từ offline sang realtime không điều khiển, sau đó mới closed-loop thật.

## Bước 1: Offline inference/eval

Mục tiêu của bước này là kiểm tra predictor và planner bằng dữ liệu đã có, chưa đụng xe thật.

Luồng chạy:

1. Load checkpoint predictor đã train.
2. Load feature cache đã trích xuất sẵn.
3. Lấy latent frame hiện tại `z_t`.
4. Lấy goal latent `z_{t+k}` từ cùng sequence trong dataset.
5. Planner sinh nhiều chuỗi action ứng viên.
6. Với từng chuỗi action, predictor rollout latent tương lai.
7. Chọn chuỗi action có latent cuối gần goal latent nhất.
8. Ghi planned action ra file để kiểm tra.

Đây là cách giống tinh thần JEPA: model không trực tiếp đoán action từ ảnh, mà planner tìm action bằng cách dùng world model dự đoán hậu quả của action.

Đầu ra cần xem:

- `planned_first_action`: action đầu tiên planner muốn chạy.
- `planned_action_sequence`: toàn bộ chuỗi action ứng viên tốt nhất.
- `planned_final_l1`: khoảng cách latent cuối khi dùng action planner.
- `groundtruth_final_l1`: khoảng cách latent cuối khi dùng action thật trong log.
- `first_action_abs_error`: sai khác giữa action đầu planner và action thật.

Không nên kỳ vọng planned action luôn trùng action thật. Vì một goal latent có thể đạt được bằng nhiều chuỗi action khác nhau. Chỉ số quan trọng hơn là planned rollout có kéo latent tới gần goal hay không.

## Bước 2: Live dry-run

Mục tiêu của bước này là chạy realtime pipeline nhưng chưa gửi lệnh xuống xe.

Luồng chạy mong muốn:

1. Nhận camera frame thật từ điện thoại hoặc camera.
2. Resize/normalize đúng như feature extractor.
3. Chạy encoder V-JEPA 2.1 để lấy latent frame hiện tại.
4. Lấy state sensor hiện tại.
5. Chọn goal latent/subgoal.
6. Planner sinh action.
7. Log action ra terminal/file/W&B.
8. Không gửi action xuống xe.

Bước này dùng để đo:

- FPS thực tế.
- Latency từ camera tới action.
- VRAM/RAM.
- Action có bị giật hoặc bão hòa không.
- Khi mất frame/sensor thì pipeline có fallback đúng không.

Nếu dry-run chưa ổn thì tuyệt đối chưa nên bật closed-loop.

## Bước 3: Closed-loop thật

Closed-loop chỉ nên bật khi offline và dry-run đã ổn.

Các điều kiện tối thiểu:

- Có giới hạn throttle an toàn, ví dụ chỉ cho chạy rất chậm lúc đầu.
- Có clamp steering/throttle.
- Có watchdog: quá một khoảng thời gian không nhận action mới thì xe về neutral.
- Có neutral fallback khi model/planner lỗi.
- Có công tắc dừng khẩn cấp.
- Có log đầy đủ frame timestamp, state, action planned, action sent.
- Có kiểm tra action scale từ float sang byte/servo command.

Luồng closed-loop:

1. Camera/sensor vào PC.
2. PC encode frame bằng đúng encoder/checkpoint.
3. Planner sinh action raw.
4. Safety layer clamp action.
5. Gửi action xuống xe qua kênh điều khiển.
6. Nếu lỗi hoặc timeout thì gửi neutral.

## Không được trộn format JEPA và NN-JEPA

Đây là điểm quan trọng nhất khi triển khai inference.

JEPA mới của bạn dùng hướng `VJEPA2ACCar`:

- Encoder: V-JEPA 2.1 ViT-L.
- Ảnh: 256px trong pipeline hiện tại của JEPA.
- Token/frame: `256`.
- Latent dim: `1024`.
- State: full IMU 10D, ví dụ `[speed, gx, gy, gz, ax, ay, az, rx, ry, rz]`.
- Planner: CEM + `CarDynamics`.
- Model: patch-token AC predictor riêng của JEPA.

NN-JEPA hiện tại dùng:

- Encoder: V-JEPA 2.1 ViT-B 384.
- Ảnh: 384px khi extract feature.
- Token/frame: `576`.
- Latent dim: `768`.
- State hiện tại: `[yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]`.
- Action: `[steering_cmd_t, throttle_cmd_t]`.
- Predictor: `simple` hoặc `official_lite` trong `src/models/rc_jepa_ac.py`.

Vì vậy không được:

- Load checkpoint JEPA `VJEPA2ACCar` vào model NN-JEPA.
- Dùng feature `ViT-L 256, D=1024` cho predictor NN-JEPA train với `ViT-B 384, D=768`.
- Dùng planner cần full IMU 10D cho state 5D hiện tại mà không chỉnh lại.
- Dùng action/state raw trực tiếp nếu model train bằng normalized action/state.

## Cách port ý tưởng JEPA sang NN-JEPA

Phần nên port:

- CEM planner.
- Ý tưởng score latent cuối so với goal latent.
- Clamp action.
- Penalty action magnitude.
- Penalty action smoothness.
- Log planned sequence để debug.

Phần chưa nên port nguyên xi:

- `CarDynamics` của JEPA, vì NN-JEPA hiện chưa có `speed` trong state mặc định.
- Checkpoint/model `VJEPA2ACCar`, vì shape khác.
- Live controller, vì kênh gửi lệnh phần cứng cần kiểm tra riêng.

Với state 5D hiện tại, rollout state hợp lý nhất là giống train loop hiện tại:

- Giữ các sensor tương lai chưa biết bằng giá trị state hiện tại.
- Cập nhật `steering_last_t` và `throttle_last_t` từ action ứng viên trước đó.

Cách này không mạnh bằng dynamics thật, nhưng nhất quán với cách model đang được train trong NN-JEPA.

## Kết luận triển khai

Thứ tự triển khai đúng cho NN-JEPA là:

1. Thêm offline CEM planner trên feature cache.
2. Chạy planner trên split `val/test`, ghi JSONL/CSV.
3. So sánh planned rollout với ground-truth rollout.
4. Nếu ổn, mới viết live dry-run.
5. Nếu dry-run ổn, mới viết closed-loop có safety.

Inference hiện tại nên dừng ở offline planner trước. Đây là bước đủ gần với JEPA để kiểm tra logic world model, nhưng chưa tạo rủi ro phần cứng.
