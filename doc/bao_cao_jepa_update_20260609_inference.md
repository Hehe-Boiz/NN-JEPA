# Báo cáo cập nhật JEPA sau pull mới nhất và đánh giá hướng inference

Ngày viết: 2026-06-09

Repo đang làm train chính: `NN-JEPA`

Repo tham chiếu vừa pull: `JEPA/`

## 1. Trạng thái JEPA sau pull

Repo `JEPA/` hiện đang ở:

```text
HEAD: e841729 Update HANDOFF: overnight results + inference blockers
branch: main
origin/main: e841729
```

`git status` trong `JEPA/` chỉ còn:

```text
D data/.gitkeep
```

Đây là dirty local do thư mục `data/` local, không phải thay đổi code quan trọng.

Commit kỹ thuật quan trọng ngay trước HEAD:

```text
30215d8 Add VJEPA2ACCar: faithful patch-token V-JEPA-2-AC port for RC car
```

Commit mới nhất `e841729` chủ yếu cập nhật `docs/HANDOFF.md`, ghi lại kết quả overnight + blocker inference.

## 2. Cập nhật lớn trong JEPA

JEPA đã có một hướng mới rõ ràng hơn cho control:

```text
VJEPA2ACCar
```

File chính:

```text
JEPA/src/jepa_wm/models/vjepa2_ac_car.py
JEPA/src/jepa_wm/data/ac_clip.py
JEPA/src/jepa_wm/data/state.py
JEPA/src/jepa_wm/planning/cem.py
JEPA/src/jepa_wm/planning/dynamics.py
JEPA/src/jepa_wm/engine/train_ac_car.py
JEPA/src/jepa_wm/engine/encode_patch.py
```

Tài liệu chính:

```text
JEPA/docs/HANDOFF.md
JEPA/docs/VJEPA2_AC_CAR.md
```

Config chính:

```text
JEPA/configs/model/vjepa_ac_car.yaml
JEPA/configs/model/vjepa_ac_car_minimal.yaml
JEPA/configs/model/vjepa_ac_car_residual.yaml
JEPA/configs/train/vjepa_ac_car.yaml
JEPA/configs/train/vjepa_ac_car_minimal.yaml
JEPA/configs/train/vjepa_ac_car_residual.yaml
```

## 3. Kết quả JEPA báo cáo trong HANDOFF

Theo `JEPA/docs/HANDOFF.md`, model `VJEPA2ACCar` đã train/eval offline và thắng pooled baseline.

Bảng kết quả chính:

```text
VJEPA2ACCar v1, 10-D IMU, patch:
  rollout@1 ratio = 0.826
  rollout@3 ratio = 0.775
  checkpoint = checkpoints/vjepa_ac_car/vjepa_ac_car/best.pt

vjepa_ac_pool baseline:
  rollout@1 ratio = 0.867
```

Ý nghĩa:

```text
ratio < 1.0 = tốt hơn identity baseline
0.826 tốt hơn 0.867
patch-token + state tốt hơn pooled latent baseline
```

Config v1 trong JEPA:

```text
Encoder: V-JEPA 2.1 ViT-L frozen
Input encode: 256px mỗi frame
Patch tokens: 16 x 16 = 256 token/frame
Embed dim: 1024
State: 10-D [speed, gx, gy, gz, ax, ay, az, rx, ry, rz]
Horizon train: 4
Frame stride: 2
Batch: 40
LR: 2.5e-4
Loss: L1 teacher-forcing + 2-step rollout
```

Điểm đáng chú ý:

```text
JEPA đang dùng ViT-L 256px patch-token fp16 cache
NN-JEPA hiện dùng ViT-B 384px full-token fp32 cache
```

## 4. VJEPA2ACCar khác gì NN-JEPA hiện tại

### 4.1. NN-JEPA hiện tại

NN-JEPA hiện tại:

```text
features_dir = data/processed/features/vjepa2_1_vitb_384_ema_fp32
encoder = V-JEPA 2.1 ViT-B 384
tokens_per_frame = 576
embed_dim = 768
raw_frames_per_sample = 8
state = [yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]
action = [steering_cmd_t, throttle_cmd_t]
predictor = simple hoặc official_lite
```

Hiện NN-JEPA train/eval chủ yếu là:

```text
world-model loss
teacher_forcing_loss
rollout_loss
offline eval/infer latent prediction
```

NN-JEPA chưa có:

```text
CEM planner để tự chọn action
CarDynamics để rollout state tương lai theo action
online encoder từ frame live
closed-loop inference loop
đường gửi action thật về xe
```

### 4.2. JEPA VJEPA2ACCar

JEPA mới có hướng gần inference hơn:

```text
patch-token predictor
state token full IMU + speed
CEMPlannerAC
CarDynamics
action clamp cho throttle an toàn
```

Khác biệt kỹ thuật chính:

| Mục | NN-JEPA hiện tại | JEPA VJEPA2ACCar |
|---|---|---|
| Encoder cache | ViT-B 384 | ViT-L 256 |
| Token/frame | 576 | 256 |
| Embed dim | 768 | 1024 |
| State | 5D IMU/action-last | 10D speed + full IMU |
| Predictor | simple/official_lite | VJEPA2ACCar patch-token |
| Planning | chưa có | CEMPlannerAC |
| State rollout | copy action-last đơn giản | CarDynamics speed/yaw |
| Output inference | latent error | action `[steer, throttle]` |

## 5. Inference kiểu JEPA là gì

Theo `JEPA/docs/HANDOFF.md`, inference mong muốn là:

```text
phone TCP stream JPEG + meta
  -> PC nhận frame
  -> PC encode V-JEPA online
  -> lấy goal/subgoal latent
  -> CEMPlannerAC chọn action
  -> gửi action về xe
```

Luồng cụ thể:

```text
current frame
  -> V-JEPA encoder
  -> patch tokens z0

current state
  -> speed + IMU

goal image/subgoal image
  -> V-JEPA encoder
  -> goal patch tokens

CEM:
  sample nhiều chuỗi action
  dùng CarDynamics rollout state tương lai
  dùng VJEPA2ACCar rollout patch tokens tương lai
  score = L1(predicted final patch, goal patch)
  lấy action đầu tiên của sequence tốt nhất
```

Đây là đúng hướng nếu mục tiêu là xe tự lái thật, vì nó biến world model thành controller.

## 6. Tôi thấy hướng inference kiểu này thế nào?

Kết luận ngắn:

```text
Nên làm, nhưng không nên nhảy thẳng vào closed-loop thật.
```

Lý do nên làm:

```text
1. Nó đúng bản chất world model: model dự đoán tương lai, planner chọn action.
2. Nó tạo output action thật [steer, throttle], không chỉ metric latent.
3. CEMPlannerAC + CarDynamics trong JEPA đã là khung hợp lý để port.
4. VJEPA2ACCar đã có kết quả offline tốt hơn pooled baseline.
5. Đây là bước biến train/eval thành tự lái thật.
```

Lý do không nên nhảy thẳng vào xe chạy thật:

```text
1. JEPA chưa có scripts/inference_loop.py hoàn chỉnh.
2. pc_stream_view.py hiện chỉ nhận frame phone -> PC, chưa gửi action ngược.
3. Android app chưa relay action PC -> phone -> ESP32.
4. controller.py trong JEPA vẫn là UDP cũ, chưa đổi sang dongle serial.
5. Closed-loop thật cần watchdog, emergency stop, clamp throttle, mode AUTO rõ ràng.
```

Vì vậy hướng đúng là:

```text
offline planner first
then live dry-run no-control
then closed-loop low-speed
```

## 7. Blocker inference thật hiện tại

JEPA tự ghi trong `HANDOFF.md`:

```text
scripts/inference_loop.py: chưa có
phone TCP stream: hiện chỉ phone -> PC
PC -> phone -> ESP32: chưa có
robot/capture/controller.py: còn UDP cũ
controller throttle map: còn logic cũ, cần Mode-3 linear map
```

Trong `JEPA/CLAUDE.md` cũng ghi:

```text
PC -> car nên đi qua dongle serial ESP-NOW
payload 2 byte [steer, throttle]
serial line = hex + '\n'
```

Mapping đúng:

```text
steer byte:
  0 = full left
  127 = center
  255 = full right

throttle byte:
  0 = full reverse
  127 = neutral
  255 = full forward

float [-1, 1] -> byte = int((value + 1.0) / 2.0 * 255)
```

Safety clamp nên dùng:

```text
steering: [-1, 1]
throttle: [-0.16, 0.15]
```

## 8. Hướng triển khai inference nên làm trong NN-JEPA

Tôi đề xuất không copy nguyên repo JEPA sang NN-JEPA ngay. Nên port có kiểm soát theo 4 mốc.

### Mốc 1: Offline planner/eval trên data đã có

Mục tiêu:

```text
chứng minh planner chọn action hợp lý trên val/test trước khi đụng xe thật
```

Việc cần làm:

```text
1. Thêm CarDynamics tương tự JEPA.
2. Thêm CEM planner cho predictor hiện tại của NN-JEPA.
3. Viết eval_goal_reaching_features.py:
   - lấy window trong val/test
   - goal = latent frame tương lai
   - CEM chọn action để tới goal
   - so sánh với random và action thật
```

Metric nên log:

```text
CEM final latent L1
teacher action final latent L1
random mean/best latent L1
CEM/random ratio
|steer_cem - steer_true|
|throttle_cem - throttle_true|
```

Đây là bước quan trọng nhất trước khi inference thật.

### Mốc 2: Online encoder smoke test

Mục tiêu:

```text
1 frame live -> V-JEPA encoder -> feature đúng shape -> planner chạy được
```

Với NN-JEPA hiện tại:

```text
encoder = ViT-B 384
output = [576, 768]
```

Cần đảm bảo:

```text
preprocess live frame giống extract feature offline
resize 384
ImageNet norm
checkpoint key ema_encoder
dtype thống nhất
```

### Mốc 3: Live inference dry-run, chưa gửi action

Mục tiêu:

```text
phone stream -> PC nhận frame -> encode -> planner -> in action ra terminal
```

Không gửi action về xe.

Log cần có:

```text
fps encode
planner latency
chosen steer/throttle
score CEM
current state
goal/subgoal id
```

Nếu latency quá cao thì chưa chạy thật.

### Mốc 4: Closed-loop thật với safety

Chỉ chạy khi 3 mốc trên ổn.

Yêu cầu bắt buộc:

```text
manual kill switch / mode AUTO rõ ràng
watchdog: nếu PC mất command > 500ms -> neutral
throttle clamp [-0.16, 0.15]
rate limit action để tránh giật
log toàn bộ frame/action/state
```

Nên chạy đầu tiên:

```text
goal rất gần
throttle rất thấp
không có vật cản
người cầm điều khiển sẵn để override
```

## 9. Nên port JEPA VJEPA2ACCar vào NN-JEPA không?

Có, nhưng theo thứ tự.

Ưu tiên hiện tại của NN-JEPA:

```text
1. Hoàn tất train tiny/simple hiện tại để có baseline ổn.
2. Thêm offline CEM eval cho checkpoint hiện tại.
3. Nếu offline CEM tốt, viết live dry-run.
4. Sau đó mới port VJEPA2ACCar patch-token 256/ViT-L hoặc train lại official-style lớn hơn.
```

Không nên làm ngay:

```text
nhảy thẳng sang closed-loop thật khi chưa có offline planner metric
copy toàn bộ JEPA inference mà chưa khớp feature format
mix checkpoint ViT-B 384 NN-JEPA với planner/code giả định ViT-L 256 JEPA
```

Lý do:

```text
NN-JEPA feature = 576 token/frame, D=768
JEPA VJEPA2ACCar = 256 token/frame, D=1024
Hai checkpoint/predictor không dùng lẫn trực tiếp được.
```

## 10. Đánh giá nếu dùng model NN-JEPA hiện tại để inference

Có thể triển khai CEM trên model hiện tại, nhưng cần hiểu giới hạn.

Model hiện tại có:

```text
state 5D = yaw_rate, accel_x, accel_y, steering_last, throttle_last
action 2D = steering_cmd, throttle_cmd
predictor latent tokens
```

Điểm yếu:

```text
không có speed đáng tin
state rollout tương lai hiện chỉ copy action-last, chưa có dynamics vật lý
planner sẽ tối ưu trong latent space nhưng state tương lai chưa chuẩn
```

Cách làm tạm:

```text
CarDynamics đơn giản:
  yaw_rate_next ~= k_yaw * steer * throttle_or_speed_proxy
  accel_x_next ~= k_thr * throttle - k_drag * accel/speed_proxy
  steering_last_next = steering_cmd
  throttle_last_next = throttle_cmd
```

Nhưng nếu không có speed thật thì planner sẽ yếu hơn JEPA `VJEPA2ACCar`.

Khuyến nghị:

```text
Nếu mục tiêu inference thật, nên đưa speed hoặc speed proxy vào state.
```

Nguồn speed có thể là:

```text
GPS speed nếu chạy ngoài trời
wheel encoder nếu lắp được
visual odometry / optical flow estimate nếu làm sau
IMU integration ngắn hạn nếu đã lọc tốt
```

## 11. Khuyến nghị cuối

Tôi đánh giá hướng inference kiểu JEPA là đúng và nên làm, nhưng theo thứ tự này:

```text
1. Giữ NN-JEPA train baseline hiện tại chạy xong.
2. Port CEM offline eval trước, chưa điều khiển xe.
3. Nếu CEM/random tốt và action recovery ổn, viết live dry-run.
4. Sau đó mới thêm controller serial/dongle và closed-loop thật.
5. Song song, cân nhắc port VJEPA2ACCar hoặc train lại NN-JEPA theo patch-token + full IMU/speed.
```

Việc gần nhất nên làm trong NN-JEPA:

```text
tools/eval_goal_reaching_features.py
models/planning.py hoặc tools/cem_planner_features.py
```

Đây là bước cầu nối giữa train world model và xe tự lái thật.

