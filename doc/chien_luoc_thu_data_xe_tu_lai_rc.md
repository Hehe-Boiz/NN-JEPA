# Chiến lược thu data cho xe đua đồ chơi gắn camera

Mục tiêu của tài liệu này là thiết kế một chiến lược thu data chi tiết cho xe đua đồ chơi gắn camera, phục vụ việc train model tự lái theo hướng ban đầu:

```python
video_history + state_history
→ steering_cmd / target_speed hoặc throttle_cmd
```

Dữ liệu dự kiến log:

```python
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

Hướng triển khai chính: **behavior cloning trước**, sau đó mới mở rộng sang pseudo-teacher hoặc trajectory planner nếu có đủ dữ liệu pose/map.

---

## 0. Nguyên tắc quan trọng nhất

Không nên thu data kiểu:

```text
ra bãi trống → chạy vòng vòng → lưu video/action
```

Cách đó dễ tạo dataset nghèo, model học yếu.

Nên thu theo kiểu:

```text
tạo nhiều tình huống có chủ đích
→ mỗi tình huống có mục tiêu rõ
→ log đầy đủ timestamp, frame, state, action
→ kiểm tra data ngay sau mỗi session
→ train thử
→ xem model yếu ở đâu
→ quay lại thu đúng chỗ yếu đó
```

Nói ngắn gọn:

> **Không thu data ngẫu nhiên. Thu data như đang thiết kế bài kiểm tra cho model.**

---

## 1. Trước khi thu data: xác định task

Bạn cần định nghĩa rõ xe tự lái là tự lái theo cái gì.

Với bãi đất trống/công viên, task đầu tiên nên là:

```text
Xe chạy theo track được đánh dấu bằng cone/vạch/dây/phấn.
```

Không nên bắt đầu bằng “xe tự chạy vòng vòng trong môi trường trống”, vì camera không có tín hiệu rõ để biết phải đi đâu.

Task tốt nhất ban đầu:

```text
Nhìn camera trước + trạng thái xe
→ điều khiển steering để bám track
→ điều khiển speed để vào cua chậm, ra cua nhanh
```

---

## 2. Setup môi trường thu data

### 2.1. Nơi thu data

Ưu tiên:

```text
bãi đất trống riêng
sân rộng ít người
bãi gửi xe trống
sân trường/sân thể thao khi được phép
khu công viên rất vắng và có vùng riêng
```

Không nên dùng:

```text
đường công cộng
vỉa hè đông người
công viên có trẻ em/chó/người đi bộ gần đó
khu có xe máy/xe đạp chạy ngang
```

Dù đang manual control để thu data, xe đua RC vẫn có thể nguy hiểm nếu mất tín hiệu hoặc throttle kẹt.

---

### 2.2. Đánh dấu track

Bạn nên dùng một trong các cách sau:

```text
cone nhỏ
chai nước
băng keo màu
dây màu
phấn vẽ nền
vạch giấy/cardboard
hộp carton nhỏ
```

Ban đầu nên có **tín hiệu thị giác rõ** cho camera:

```text
hai hàng cone làm biên trái/phải
hoặc một vạch centerline màu nổi
hoặc tape màu tạo đường đi
```

Không nên dùng track quá mơ hồ, vì model sẽ không biết nhìn vào đâu.

---

## 3. Những biến bắt buộc phải log

Mỗi frame nên có một dòng log.

```csv
timestamp,frame_path,v,yaw_rate,accel_x,accel_y,steering_last,throttle_last,steering_cmd,throttle_cmd,mode,session_id,scenario_tag
```

Ví dụ:

```csv
0.000,frames/000001.jpg,0.32,0.01,0.02,0.00,0.00,0.18,0.02,0.20,manual,session_001,normal_oval
0.050,frames/000002.jpg,0.34,0.02,0.03,0.01,0.02,0.20,0.05,0.22,manual,session_001,normal_oval
```

Nên có thêm:

```text
battery_voltage
fps_camera
control_rate
camera_exposure/brightness nếu lấy được
manual/autonomous/override mode
track_id
lap_id
scenario_tag
```

`scenario_tag` rất quan trọng. Nó giúp biết đoạn đó là:

```text
normal
left_curve
right_curve
recovery_left
recovery_right
slalom
hard_brake
low_light
rough_surface
```

Sau này khi model chạy tệ, bạn dễ tra lại thiếu data ở tình huống nào.

---

## 4. Đồng bộ timestamp là bắt buộc

Đừng chỉ lưu frame và action rời rạc.

Bạn cần lưu timestamp cho:

```text
camera frame
IMU/state
control command
```

Vì thực tế có latency:

```text
camera latency
WiFi latency
model inference latency
motor response latency
người lái phản ứng trễ
```

Khi train, không nên mặc định:

```text
frame_t → action_t
```

Thường nên dùng:

```text
frame_t, state_t → action_{t+Δ}
```

Ví dụ nếu camera/control ở 20 Hz:

```text
Δ = 1 frame  ≈ 50 ms
Δ = 2 frames ≈ 100 ms
```

Dataset sample nên tạo kiểu:

```python
history_len = 5
delay = 2
horizon = 5

X_t = {
  frames[t-4:t],
  states[t-4:t]
}

Y_t = {
  actions[t+2:t+7]
}
```

Nghĩa là:

```text
nhìn 0.25 giây quá khứ
dự đoán bắt đầu từ 0.1 giây tương lai
dự đoán 0.25 giây tiếp theo
```

---

## 5. Target nên thu để train

Bạn có thể train output:

```python
[steering_cmd, throttle_cmd]
```

Nhưng hướng tốt hơn là:

```python
[steering_cmd, target_speed]
```

Sau đó dùng PID để đổi `target_speed` thành `throttle_cmd`.

Lý do: ngoài trời throttle bị ảnh hưởng bởi:

```text
pin yếu/mạnh
mặt đất xi măng/đất/cỏ/sỏi
độ dốc
bụi
ma sát bánh
trượt bánh
```

Cùng một `throttle_cmd = 0.3` nhưng vận tốc thật có thể rất khác nhau.

Còn `target_speed` có ý nghĩa ổn định hơn:

```text
vào cua → target_speed thấp
đường thẳng → target_speed cao hơn
gần mất lái → target_speed giảm
```

Nếu chưa làm PID speed được, giai đoạn đầu vẫn có thể train throttle trực tiếp, nhưng nên chuẩn bị chuyển sang target speed.

---

## 6. Các loại tình huống cần thu

Đây là phần quan trọng nhất.

---

### Nhóm A — Chạy chuẩn

Mục tiêu: dạy model hành vi bình thường.

Thu các đoạn:

```text
chạy giữa track
giữ lane ổn định
vào cua đúng
ra cua đúng
giữ tốc độ đều
không lắc lái
```

Track nên có:

```text
đường thẳng
cua trái
cua phải
oval
số 8
S-curve
slalom nhẹ
```

Tỷ lệ đề xuất:

```text
40–50% dataset ban đầu
```

---

### Nhóm B — Recovery lệch trái/phải

Đây là nhóm cực kỳ quan trọng.

Bạn cần cố tình đưa xe vào trạng thái lệch rồi **điều khiển kéo về đúng track**.

Tình huống:

```text
xe lệch trái nhẹ → kéo về giữa
xe lệch trái nhiều → kéo về giữa
xe lệch phải nhẹ → kéo về giữa
xe lệch phải nhiều → kéo về giữa
xe gần ra khỏi track → kéo về
xe vào cua quá rộng → sửa lại
xe vào cua quá hẹp → sửa lại
xe ra cua bị lệch → sửa lại
```

Nhưng có một bẫy rất lớn:

> Khi bạn cố tình đánh lái sai để đưa xe lệch ra ngoài, đoạn đó không nên dùng như label “đúng”.

Ví dụ:

```text
Bạn cố tình đánh lái trái để làm xe lệch trái.
Đoạn đánh lái trái đó là hành vi xấu.
Nếu train vào, model sẽ học cách tự làm xe lệch.
```

Cách xử lý:

```text
Gắn tag: induce_error
Gắn tag: recovery
Chỉ dùng đoạn recovery để train chính.
```

Ví dụ một sequence:

```text
0–2s: chạy bình thường        → dùng train
2–3s: cố tình làm xe lệch     → không dùng hoặc tag induce_error
3–6s: kéo xe về giữa          → dùng train mạnh
```

Nên log `scenario_tag`:

```text
induce_left_error
recover_from_left
induce_right_error
recover_from_right
```

Tỷ lệ đề xuất:

```text
25–35% dataset ban đầu
```

Nếu model hay văng khỏi track, tăng nhóm này lên.

---

### Nhóm C — Nhiều tốc độ

Cùng một cảnh nhưng tốc độ khác nhau cần action khác nhau.

Thu:

```text
chạy rất chậm
chạy chậm
chạy vừa
chạy nhanh vừa phải
giảm tốc trước cua
tăng tốc sau cua
giữ tốc độ ổn định trên đường thẳng
```

Tình huống cụ thể:

```text
đường thẳng tốc độ thấp
đường thẳng tốc độ vừa
cua trái tốc độ thấp
cua trái tốc độ vừa
cua phải tốc độ thấp
cua phải tốc độ vừa
vào cua hơi nhanh rồi giảm tốc
ra cua rồi tăng tốc
```

Không nên thu tốc độ quá cao giai đoạn đầu. Model mới train rất dễ sai.

Tỷ lệ đề xuất:

```text
10–20% dataset
```

---

### Nhóm D — Cua trái/cua phải đủ loại

Bạn cần thu riêng nhiều kiểu cua, vì xe đua đồ chơi thường fail ở cua.

Tình huống:

```text
cua trái rộng
cua phải rộng
cua trái hẹp
cua phải hẹp
cua chữ U
cua 90 độ
cua liên tiếp trái-phải
cua liên tiếp phải-trái
S-curve
số 8
chicane
slalom cone
```

Mỗi cua nên thu:

```text
vào cua chậm
vào cua vừa
vào cua hơi trễ rồi sửa
vào cua hơi sớm rồi sửa
ra cua rộng rồi sửa
ra cua hẹp rồi sửa
```

Đừng chỉ thu một kiểu cua đẹp.

---

### Nhóm E — Đường thẳng

Nghe đơn giản nhưng vẫn cần.

Thu:

```text
đường thẳng dài
đường thẳng ngắn
đường thẳng sau cua
đường thẳng trước cua
đang lệch trái trên đường thẳng → kéo về
đang lệch phải trên đường thẳng → kéo về
đường thẳng tốc độ thấp
đường thẳng tốc độ vừa
```

Model cần học rằng trên đường thẳng thì steering nên ổn định, không lắc trái phải.

---

### Nhóm F — Thay đổi ánh sáng

Ngoài trời ánh sáng thay đổi rất mạnh.

Thu:

```text
nắng trực tiếp
bóng râm
nửa nắng nửa râm
chiều mát
trời hơi âm u
mặt đường phản sáng
camera nhìn ngược sáng nhẹ
```

Không nên thu trong điều kiện quá nguy hiểm như quá tối hoặc chói hoàn toàn, trừ khi muốn model xử lý và có cơ chế dừng.

Nên thêm tag:

```text
sunny
shadow
mixed_shadow
overcast
low_light
glare
```

Quan trọng: nếu camera bị chói đến mức người cũng khó nhìn, đó nên là tình huống **failsafe dừng xe**, không phải ép model lái tiếp.

---

### Nhóm G — Mặt đường khác nhau

Xe đồ chơi rất nhạy với mặt đường.

Thu:

```text
xi măng nhẵn
gạch
nhựa đường
đất cứng
đất bụi
nền hơi gồ
nền có sỏi nhỏ
nền có lá cây ít
```

Không nên test trên:

```text
cỏ dày
sỏi lớn
mặt quá trơn
vũng nước
đường có người
```

Mỗi mặt đường ảnh hưởng đến:

```text
v_t
accel_x
accel_y
yaw_rate
throttle response
độ trượt bánh
```

Vì có IMU/state, đây là dữ liệu rất hữu ích.

---

### Nhóm H — Camera rung, blur, nhiễu

Ngoài trời xe rung nhiều.

Nên thu một ít:

```text
camera rung nhẹ
ảnh hơi blur do tốc độ
mặt đường gồ nhẹ
xe qua đoạn xóc nhỏ
```

Nhưng không nên cố tình làm camera lỏng. Camera mount phải chắc. Dữ liệu rung quá nặng làm model học khó và thực tế cũng nguy hiểm.

Nên có failsafe:

```text
nếu ảnh quá blur / camera lost / frame timeout → throttle = 0
```

---

### Nhóm I — Obstacle tĩnh đơn giản

Chỉ làm sau khi xe đã chạy track ổn.

Obstacle nên là vật mềm/nhẹ:

```text
cone
hộp carton
chai nhựa rỗng
miếng xốp
```

Tình huống:

```text
obstacle nằm bên trái track
obstacle nằm bên phải track
obstacle gần giữa track nhưng còn đường né
obstacle sau cua
obstacle trên đường thẳng
```

Nhưng cần xác định task rõ:

```text
né obstacle rồi quay lại track
```

Nếu không, model có thể học lẫn giữa “bám track” và “né vật”.

Giai đoạn đầu, chưa cần obstacle động.

---

### Nhóm J — Stop / giảm tốc

Bạn cần data để xe biết giảm tốc hoặc dừng.

Tình huống:

```text
đến cuối track → giảm tốc
gần obstacle → giảm tốc
vào cua gắt → giảm tốc
mất track → giảm tốc/dừng
người điều khiển nhả throttle → xe dừng
```

Nếu output là target speed, nhóm này rất quan trọng.

Tag:

```text
slow_down
stop
hard_brake
coast
```

---

### Nhóm K — Mất track / không chắc chắn

Nên thu một ít tình huống “không nên lái tiếp”.

Ví dụ:

```text
track biến mất khỏi camera
camera bị che một phần
quá gần mép track
xe quay sai hướng
ảnh quá tối
ảnh quá chói
```

Target ở đây nên là:

```text
steering về trung tính
target_speed = 0
throttle = 0
```

Đây là data cho failsafe hoặc head phụ “uncertain/stop”.

Nếu chỉ behavior cloning steering/throttle bình thường, có thể tách nhóm này ra làm rule-based safety thay vì train chung.

---

## 7. Các track nên dựng để thu data

### Track 1: Oval đơn giản

Mục tiêu:

```text
chạy ổn định
vào cua trái/phải cơ bản
giữ tốc độ
```

Biến thể:

```text
oval rộng
oval hẹp
oval dài
oval ngắn
```

Thu:

```text
10 vòng chậm
10 vòng vừa
nhiều recovery trái/phải
```

---

### Track 2: Hình số 8

Mục tiêu:

```text
có cả cua trái và cua phải liên tục
xe học chuyển hướng
học tránh lắc lái
```

Track số 8 rất tốt vì tránh model chỉ quen một chiều cua.

Thu:

```text
chạy thuận
chạy ngược
chạy chậm
chạy vừa
recovery ở điểm giao giữa số 8
```

---

### Track 3: S-curve

Mục tiêu:

```text
học chuyển cua trái-phải
học trả lái
học giữ tốc độ khi đổi hướng
```

Tình huống:

```text
S rộng
S hẹp
S dài
S ngắn
```

---

### Track 4: Slalom cone

Mục tiêu:

```text
học phản ứng nhanh nhưng mượt
học tránh over-steering
```

Ban đầu đặt cone thưa, sau đó mới hẹp dần.

---

### Track 5: Long straight + hairpin

Mục tiêu:

```text
đường thẳng tăng tốc
trước cua giảm tốc
cua chữ U
ra cua tăng tốc
```

Cực kỳ hữu ích nếu muốn xe chạy kiểu racing hơn.

---

### Track 6: Random cone corridor

Tạo hành lang bằng cone:

```text
rộng → hẹp
hẹp → rộng
cua nhẹ
cua gắt
đoạn lệch trái/phải
```

Mục tiêu: model không chỉ memorize một track cố định.

---

## 8. Chiến lược thu theo buổi

Một buổi thu data nên có cấu trúc.

### Buổi 1 — Kiểm tra hệ thống

Mục tiêu: không phải thu nhiều, mà kiểm tra log đúng.

Làm:

```text
chạy 3–5 phút
kiểm tra frame có lưu đủ không
kiểm tra timestamp tăng đều không
kiểm tra action có đúng không
kiểm tra steering dấu trái/phải có đúng không
kiểm tra IMU có giá trị hợp lý không
kiểm tra video khớp action không
```

Không nên thu 1 tiếng rồi mới phát hiện timestamp/action lệch.

---

### Buổi 2 — Dataset cơ bản

Track: oval + số 8.

Thu:

```text
10 phút chạy chuẩn chậm
10 phút chạy chuẩn vừa
10 phút recovery trái/phải
5 phút vào cua trễ/sớm rồi sửa
5 phút điều kiện ánh sáng khác
```

Mục tiêu: train model đầu tiên.

---

### Buổi 3 — Bổ sung điểm yếu

Sau khi train model đầu tiên, xem nó yếu ở đâu:

```text
hay văng ở cua trái?
hay lắc trên đường thẳng?
hay ra cua quá rộng?
hay tốc độ quá cao?
hay mất track khi có bóng râm?
```

Buổi này chỉ thu đúng các lỗi đó.

---

### Buổi 4 — Autonomous + override

Cho model chạy rất chậm.

Bạn luôn cầm remote.

Khi model sai:

```text
ngắt autonomous
manual kéo về
log đoạn override/recovery
```

Tag:

```text
autonomous
override
recovery_after_override
```

Đây là data rất quý vì nó là lỗi thật của model.

---

## 9. Tỷ lệ dataset khuyến nghị

Dataset ban đầu:

```text
40% chạy chuẩn
30% recovery
15% cua khó / track phức tạp
10% tốc độ và mặt đường khác nhau
5% stop/failsafe/ánh sáng khó
```

Sau khi model chạy được:

```text
30% chạy chuẩn
40% lỗi thật + recovery
20% track mới / điều kiện mới
10% stop/failsafe
```

Điểm quan trọng:

> **Model yếu ở đâu thì dataset tiếp theo phải tập trung vào đó.**

Không cần cứ thu đều tất cả.

---

## 10. Những tình huống cụ thể nên thu

### Chạy thẳng

```text
straight_center_slow
straight_center_medium
straight_left_offset_recover
straight_right_offset_recover
straight_after_left_curve
straight_after_right_curve
straight_before_left_curve
straight_before_right_curve
straight_with_shadow
straight_with_rough_surface
```

### Cua trái

```text
left_curve_wide_slow
left_curve_wide_medium
left_curve_tight_slow
left_curve_tight_medium
left_curve_enter_too_early_recover
left_curve_enter_too_late_recover
left_curve_exit_wide_recover
left_curve_exit_inside_recover
left_curve_with_shadow
left_curve_on_rough_surface
```

### Cua phải

```text
right_curve_wide_slow
right_curve_wide_medium
right_curve_tight_slow
right_curve_tight_medium
right_curve_enter_too_early_recover
right_curve_enter_too_late_recover
right_curve_exit_wide_recover
right_curve_exit_inside_recover
right_curve_with_shadow
right_curve_on_rough_surface
```

### S-curve

```text
s_curve_left_to_right_slow
s_curve_right_to_left_slow
s_curve_left_to_right_medium
s_curve_right_to_left_medium
s_curve_late_transition_recover
s_curve_oversteer_recover
```

### Slalom

```text
slalom_wide_spacing
slalom_medium_spacing
slalom_slow
slalom_medium
slalom_missed_cone_recover
```

### Recovery

```text
recover_from_left_small
recover_from_left_large
recover_from_right_small
recover_from_right_large
recover_from_near_boundary_left
recover_from_near_boundary_right
recover_after_spin_small
recover_after_wrong_heading
recover_after_understeer
recover_after_oversteer
```

### Speed

```text
constant_low_speed
constant_medium_speed
accelerate_on_straight
decelerate_before_curve
slow_in_curve
accelerate_after_curve
stop_at_end
emergency_slowdown
```

### Surface / lighting

```text
sunny_concrete
shadow_concrete
mixed_shadow
overcast
late_afternoon
slightly_rough_ground
dusty_ground
brick_surface
asphalt_surface
```

### Failure / stop

```text
track_lost_stop
camera_blur_stop
too_close_to_boundary_stop
wrong_direction_stop
signal_lost_stop
low_battery_slow
```

---

## 11. Một lỗi rất hay gặp: data bị lệch phân phối

Nếu chạy một track oval một chiều quá nhiều, dataset sẽ bị lệch.

Ví dụ:

```text
80% cua trái
20% cua phải
```

Model sẽ giỏi rẽ trái nhưng tệ rẽ phải.

Nên luôn kiểm tra phân phối:

```text
bao nhiêu frame đường thẳng?
bao nhiêu frame cua trái?
bao nhiêu frame cua phải?
bao nhiêu frame recovery?
bao nhiêu frame tốc độ thấp/vừa?
bao nhiêu frame nắng/râm?
```

Một dataset tốt không phải dataset to nhất, mà là dataset có đủ tình huống cần thiết.

---

## 12. Không nên dùng toàn bộ data để train

Cần chia:

```text
train
validation
test
```

Nhưng không chia random từng frame, vì frame liên tiếp rất giống nhau. Nếu random từng frame, validation sẽ quá dễ và đánh lừa bạn.

Nên chia theo session hoặc theo track:

```text
train:
  session_001, 002, 003

validation:
  session_004

test:
  session_005 ở track hơi khác
```

Tốt nhất:

```text
validation/test phải là buổi chạy khác, ánh sáng khác, track hơi khác
```

Như vậy mới biết model có generalize không.

---

## 13. Kiểm tra chất lượng data sau mỗi buổi

Sau mỗi session, nên kiểm tra ngay.

### 13.1. Kiểm tra frame

```text
frame có bị mất không?
FPS có đều không?
ảnh có quá tối/chói không?
camera có bị lệch không?
camera có rung quá không?
```

### 13.2. Kiểm tra action

```text
steering_cmd có đúng dấu không?
throttle_cmd có bị kẹt không?
giá trị có nằm trong range không?
có spike bất thường không?
```

Ví dụ:

```text
steering_cmd nằm trong [-1, 1]
throttle_cmd nằm trong [0, 1] hoặc [-1, 1] tùy hệ của bạn
```

### 13.3. Kiểm tra state

```text
v_t có âm bất thường không?
yaw_rate_t có spike quá lớn không?
accel_x/y có bị lệch bias không?
IMU trục x/y có đúng hướng không?
```

### 13.4. Kiểm tra đồng bộ

Mở video replay và overlay:

```text
steering_cmd
throttle_cmd
v_t
yaw_rate_t
```

Bạn phải thấy:

```text
khi xe rẽ trái → steering plot rẽ trái
khi xe tăng tốc → throttle tăng trước/song song với v tăng
khi xe cua → yaw_rate đổi đúng chiều
```

Nếu plot không khớp video, có thể timestamp bị lệch.

---

## 14. Quy tắc loại bỏ data xấu

Không phải data nào cũng nên train.

Nên loại hoặc tag riêng:

```text
frame mất
ảnh đen
ảnh quá chói
camera bị lệch mount
xe bị lật
mất tín hiệu
người bế xe lên
xe đứng yên nhưng throttle/action vẫn ghi lỗi
đoạn bạn cố tình lái sai để tạo lệch
đoạn đâm cone mạnh
đoạn người/động vật xuất hiện gần xe
```

Một số đoạn không nên xóa hẳn mà nên tag:

```text
failure
induce_error
collision
camera_bad
```

Sau này có thể dùng để train failsafe, nhưng không nên dùng chung với data lái tốt.

---

## 15. Chiến lược augmentation

Sau khi có data thật, có thể augmentation:

```text
brightness tăng/giảm
contrast thay đổi
motion blur nhẹ
Gaussian noise nhẹ
crop/resize nhẹ
dịch ảnh trái/phải
xoay nhẹ
```

Nhưng augmentation phải cẩn thận với label steering.

### 15.1. Brightness/contrast/blur

Các augmentation này thường không cần chỉnh steering label:

```text
ảnh sáng hơn/tối hơn → steering giữ nguyên
blur nhẹ → steering giữ nguyên
noise nhẹ → steering giữ nguyên
```

### 15.2. Dịch ảnh trái/phải

Cần chỉnh steering label.

Ví dụ quy ước:

```text
steering > 0 là rẽ phải
steering < 0 là rẽ trái
```

Nếu ảnh bị dịch sao cho xe trông như lệch trái, label phải sửa về phải.

Công thức đơn giản:

```python
steering_aug = steering_original + k * shift_normalized
```

Trong đó:

```text
shift_normalized nằm khoảng [-1, 1]
k khoảng 0.1–0.3 tùy xe
```

Phải test dấu thật kỹ bằng replay trực quan.

---

## 16. Dữ liệu cho throttle/target speed

Nếu train throttle trực tiếp, cần thu:

```text
giữ throttle ổn định
tăng throttle trên đường thẳng
giảm throttle trước cua
nhả throttle khi sắp ra khỏi track
phanh/dừng nếu có brake
```

Nếu train target speed, tạo label:

```python
target_speed_t = v_{t+Δ}
```

hoặc smooth hơn:

```python
target_speed_t = average(v_{t+Δ : t+Δ+H})
```

Model học:

```text
cảnh này nên muốn tốc độ bao nhiêu
```

Sau đó PID xử lý throttle.

---

## 17. Nên thu bao nhiêu data?

Không có con số tuyệt đối, nhưng để prototype:

```text
30–60 phút data sạch có chủ đích: có thể train bản đầu
2–5 giờ data đa dạng: bắt đầu ổn hơn
nhiều ngày/nhiều điều kiện: mới tương đối robust
```

Quan trọng hơn thời lượng là chất lượng:

```text
10 phút recovery tốt > 1 giờ chạy vòng vòng đẹp
```

---

## 18. Chiến lược DAgger đơn giản

Sau khi có model đầu tiên:

```text
1. chạy autonomous tốc độ thấp
2. người luôn cầm remote
3. khi model sai, override
4. lưu đoạn sai + đoạn người sửa
5. train lại
6. lặp lại
```

Tag rõ:

```text
autonomous_ok
autonomous_fail
human_override
recovery_after_override
```

Đây là cách cực kỳ hiệu quả để sửa lỗi behavior cloning.

---

## 19. Checklist trước khi ra sân thu data

Trước khi chạy:

```text
[ ] Camera mount chắc, góc nhìn không thay đổi.
[ ] Pin đủ, motor không quá nóng.
[ ] Steering trim đã cân.
[ ] Throttle neutral đúng.
[ ] IMU hoạt động.
[ ] Speed sensor hoạt động.
[ ] Timestamp log đúng.
[ ] Frame lưu được.
[ ] Action log đúng range.
[ ] Remote override hoạt động.
[ ] Kill switch hoạt động.
[ ] Mất tín hiệu thì throttle = 0.
[ ] Khu vực test không có người/trẻ em/thú cưng gần đó.
[ ] Track/cone đã dựng rõ.
```

---

## 20. Checklist sau mỗi session

Sau mỗi session 5–10 phút:

```text
[ ] Mở vài frame xem ảnh rõ không.
[ ] Kiểm tra số frame có đúng với thời lượng không.
[ ] Kiểm tra log.csv không thiếu dòng.
[ ] Vẽ steering/throttle theo thời gian.
[ ] Vẽ v/yaw_rate/accel theo thời gian.
[ ] Replay video có overlay action.
[ ] Kiểm tra steering trái/phải đúng dấu.
[ ] Kiểm tra throttle tăng thì xe tăng tốc.
[ ] Kiểm tra tag scenario đúng.
[ ] Backup dữ liệu.
```

Không nên thu tiếp hàng giờ khi chưa kiểm tra session đầu.

---

## 21. Sai lầm cần tránh

### Sai lầm 1: chỉ thu chạy đẹp

Model sẽ không biết recovery.

### Sai lầm 2: không tag đoạn cố tình lái sai

Model có thể học hành vi sai.

### Sai lầm 3: chia train/val random theo frame

Validation sẽ ảo vì frame gần nhau quá giống.

### Sai lầm 4: không đồng bộ timestamp

Model học action lệch thời gian.

### Sai lầm 5: train throttle trực tiếp quá sớm

Throttle ngoài trời nhiễu mạnh do pin/mặt đường.

### Sai lầm 6: track không rõ

Camera không có tín hiệu, model học lung tung.

### Sai lầm 7: thu một môi trường duy nhất

Model overfit vào đúng bãi/ánh sáng/track đó.

### Sai lầm 8: không có safety

Một lỗi nhỏ có thể làm xe lao ra ngoài.

---

## 22. Protocol thu data mẫu cho một buổi 60 phút

```text
00–05 phút:
  kiểm tra hệ thống, log, camera, action.

05–15 phút:
  oval chậm, chạy chuẩn.

15–25 phút:
  oval tốc độ vừa, vào cua/ra cua mượt.

25–35 phút:
  số 8, cả trái và phải.

35–45 phút:
  recovery:
    lệch trái nhỏ/lớn
    lệch phải nhỏ/lớn
    vào cua trễ/sớm rồi sửa.

45–50 phút:
  S-curve hoặc slalom.

50–55 phút:
  stop/slowdown:
    giảm tốc trước cua
    dừng ở cuối đoạn
    throttle thấp khi track không rõ.

55–60 phút:
  kiểm tra nhanh data, ghi note session.
```

Sau buổi đó, train model đầu tiên, chạy test chậm, rồi thu tiếp đúng lỗi model mắc.

---

## 23. Kết luận chiến lược

Với project này, chiến lược đúng là:

```text
1. Thu data ở bãi trống có track rõ.
2. Log frame + state + action + timestamp + scenario_tag.
3. Thu chạy chuẩn nhưng không quá nhiều.
4. Thu rất nhiều recovery data.
5. Thu đủ cua trái/phải, tốc độ, ánh sáng, mặt đường.
6. Kiểm tra data sau mỗi session ngắn.
7. Train model đầu tiên.
8. Cho chạy chậm với override.
9. Thu lỗi thật của model.
10. Train lại theo vòng lặp.
```

Câu quan trọng nhất:

> **Data tốt cho xe tự lái không phải là data xe chạy hoàn hảo, mà là data dạy xe biết phải làm gì khi nó bắt đầu sai.**
