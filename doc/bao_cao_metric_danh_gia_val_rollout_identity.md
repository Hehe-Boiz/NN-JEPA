# Báo cáo metric đánh giá val: rollout, identity baseline và ratio

Ngày cập nhật: 2026-06-10

File này giải thích chi tiết bộ metric đang dùng để đánh giá world model JEPA-AC trong NN-JEPA, đặc biệt là ba nhóm metric:

```text
rollout_l1_hk
identity_l1_hk
ratio_hk
```

Trong tên metric thật, `k` được ghi thành số horizon cụ thể:

```text
rollout_l1_h1, rollout_l1_h2, rollout_l1_h3
identity_l1_h1, identity_l1_h2, identity_l1_h3
ratio_h1, ratio_h2, ratio_h3
```

## Kết luận hiện tại

Đã thêm đánh giá rollout-vs-identity vào `val` sau mỗi epoch cho train feature-cache.

Trước thay đổi này, NN-JEPA đã có `final_eval_val/rollout_l1_h*`, `final_eval_val/identity_l1_h*`, `final_eval_val/ratio_h*` ở cuối train, sau khi load lại `best.pt`. Tức là trước đây nó chỉ giúp xem checkpoint tốt nhất cuối cùng có tốt hơn baseline đứng yên không.

Hiện tại train loop đã có thêm per-epoch val rollout eval:

```text
val/rollout_l1_h1
val/identity_l1_h1
val/ratio_h1
val/rollout_l1_h2
val/identity_l1_h2
val/ratio_h2
val/rollout_l1_h3
val/identity_l1_h3
val/ratio_h3
```

Các metric này được:

```text
1. merge vào val_metrics sau phase val thường
2. ghi vào history.json theo epoch
3. log lên W&B dưới prefix val/*
4. in ra terminal ở dòng summary cuối mỗi epoch
```

Config mixed official-lite hiện đã bật mặc định:

```yaml
train:
  val_rollout_eval_horizon: 3
  val_rollout_eval_max_batches: 256
  final_eval_horizon: 3
```

Hai config đã bật:

```text
configs/hydra/experiment/rc_jepa_official_lite_base_mix_oldservo_frame_stride2.yaml
configs/hydra/experiment/rc_jepa_official_lite_tiny_mix_oldservo_frame_stride2.yaml
```

## Dữ liệu và model đang được đánh giá

Experiment mixed servo cũ hiện tại:

```text
root: data/experiments/servo_old_mix_v1
manifest: data/experiments/servo_old_mix_v1/processed/manifests
feature: data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
session_count: 211
frame_count: 212,840
dtype feature: fp16
image_path_key: source_frame_path
encoder_preset: vitb_384
image_size: 384
patch_size: 16
tokens_per_frame: 576
embed_dim: 768
```

Manifest hiện tại:

```text
train.jsonl: 182,558 samples
val.jsonl:    30,282 samples
test.jsonl:   30,282 samples
```

Lưu ý quan trọng: `test.jsonl` hiện là alias của `val.jsonl`, không phải test split độc lập. Vì vậy metric `test/*` nếu chạy lúc này chỉ là đo lại trên val, không được xem như kết quả tổng kết cuối cùng cho báo cáo khoa học.

Với cấu hình mixed frame-stride2 hiện tại:

```text
raw_frames_per_sample = 8
frame_stride = 2
tokens_per_frame = 576
tokens/sample = 8 * 576 = 4608 token
state_dim = 5
action_dim = 2
auto_steps = 2
```

## Ý nghĩa từng metric chính

### loss

`loss` là metric chính hiện đang dùng để train, chọn `best.pt` và early stopping.

Công thức trong train loop:

```text
loss = teacher_forcing_loss + rollout_loss
```

Ý nghĩa:

```text
teacher_forcing_loss: dự đoán latent bước kế tiếp khi luôn được cấp latent thật ở quá khứ
rollout_loss: dự đoán latent tương lai khi một phần quá khứ đã là latent model tự sinh
```

`val/loss` là metric dùng để:

```text
1. quyết định checkpoint tốt nhất best.pt
2. cập nhật best/val_loss trên W&B
3. kích hoạt early stopping nếu không cải thiện đủ lâu sau warmup
```

Hiện tại early stopping không dùng `ratio_h*`; nó vẫn dùng `val/loss`. Đây là lựa chọn hợp lý cho training ổn định vì `val/loss` chạy full val loader và khớp trực tiếp objective đang tối ưu.

### teacher_forcing_loss

`teacher_forcing_loss` đo khả năng dự đoán một bước khi lịch sử trước đó là ground truth latent.

Nói đơn giản:

```text
model thấy z_t thật, action/state thật, rồi dự đoán z_{t+1}
```

Metric này thường ổn định hơn rollout vì lỗi chưa bị tích lũy nhiều bước.

### rollout_loss

`rollout_loss` đo khả năng tự hồi quy nhiều bước trong latent space.

Nói đơn giản:

```text
model dự đoán z_{t+1}
sau đó dùng chính z_{t+1} dự đoán tiếp z_{t+2}
```

Metric này quan trọng hơn cho planner/MPC vì planner sẽ dùng world model rollout nhiều action candidate, không chỉ dự đoán một bước.

## Ba metric rollout-vs-identity mới

Ba metric này không thay thế train loss. Chúng dùng để trả lời câu hỏi thực dụng hơn:

```text
World model có dự đoán tương lai tốt hơn baseline "không làm gì, giữ nguyên latent hiện tại" không?
```

Ký hiệu:

```text
z_t: latent thật tại frame đầu trong sample
z_{t+k}: latent thật tại horizon k
z_hat_{t+k}: latent model rollout dự đoán tại horizon k
```

### rollout_l1_hk

`rollout_l1_hk` là sai số L1 giữa latent model dự đoán và latent thật ở horizon `k`.

Công thức:

```text
rollout_l1_hk = mean(abs(z_hat_{t+k} - z_{t+k}))
```

Ý nghĩa:

```text
càng thấp càng tốt
đo trực tiếp model rollout có gần latent thật không
```

Ví dụ:

```text
val/rollout_l1_h1: sai số sau 1 transition
val/rollout_l1_h2: sai số sau 2 transition
val/rollout_l1_h3: sai số sau 3 transition
```

### identity_l1_hk

`identity_l1_hk` là sai số của baseline cực đơn giản: giữ nguyên latent đầu `z_t` và xem nó khác latent tương lai `z_{t+k}` bao nhiêu.

Công thức:

```text
identity_l1_hk = mean(abs(z_t - z_{t+k}))
```

Ý nghĩa:

```text
nó đo mức độ cảnh thật sự thay đổi sau k bước
nếu xe/camera gần như đứng yên thì identity_l1_hk nhỏ
nếu xe/camera thay đổi nhiều thì identity_l1_hk lớn
```

Metric này rất quan trọng vì chỉ nhìn `rollout_l1_hk` một mình có thể bị hiểu sai. Ví dụ nếu xe đứng yên, model chỉ cần dự đoán giống frame đầu cũng có `rollout_l1` thấp. Nhưng như vậy chưa chắc model học dynamics tốt.

### ratio_hk

`ratio_hk` là tỉ lệ giữa model rollout và identity baseline.

Công thức:

```text
ratio_hk = rollout_l1_hk / identity_l1_hk
```

Ý nghĩa:

```text
ratio_hk < 1.0: model tốt hơn baseline giữ nguyên latent
ratio_hk = 1.0: model ngang baseline giữ nguyên latent
ratio_hk > 1.0: model tệ hơn baseline giữ nguyên latent
```

Đây là metric dễ đọc nhất khi muốn biết world model có thật sự học dynamics không.

Ví dụ diễn giải:

```text
ratio_h1 = 0.85 nghĩa là lỗi model ở horizon 1 chỉ bằng 85% lỗi identity baseline
ratio_h3 = 0.75 nghĩa là ở horizon 3 model tốt hơn baseline khá rõ
ratio_h3 > ratio_h1 thường là dấu hiệu rollout xa khó hơn rollout gần
```

## Vì sao cần cả val loss và ratio

`val/loss` là metric đúng để train và chọn checkpoint vì nó chính là objective đang tối ưu:

```text
teacher_forcing_loss + rollout_loss
```

Nhưng `val/loss` không nói rõ model có hơn baseline đứng yên hay không. Với data xe RC trong nhà, nhiều đoạn có thể chuyển động ít, hoặc frame liên tiếp rất gần nhau. Khi đó latent tương lai gần giống latent hiện tại, một model kém vẫn có thể có loss nhìn không quá xấu.

`ratio_hk` bổ sung góc nhìn này:

```text
nếu ratio_hk < 1: model có ích hơn baseline giữ nguyên latent
nếu ratio_hk >= 1: model chưa chứng minh được học dynamics hữu ích ở horizon đó
```

Vì vậy cách đọc hiện tại nên là:

```text
1. dùng val/loss để chọn best.pt
2. dùng ratio_h1/h2/h3 để đánh giá world model có thật sự hơn identity baseline không
3. dùng rollout_l1_hk để xem sai số tuyệt đối
4. dùng identity_l1_hk để biết val split đang khó hay dễ, chuyển động nhiều hay ít
```

## Vì sao so sánh trực tiếp val_loss giữa NN-JEPA và JEPA của bạn bạn chưa đủ công bằng

Hai code có thể dùng cùng session val, nhưng cách tạo sample khác nhau làm `val_loss` không còn cùng đơn vị thực nghiệm hoàn toàn.

Bên NN-JEPA mixed hiện tại:

```text
raw_frames_per_sample = 8
frame_stride = 2
teacher-forcing transitions = 7 cặp
auto_steps = 2
tokens_per_frame = 576
embed_dim = 768
```

Bên JEPA của bạn bạn theo các lần audit gần đây:

```text
horizon = 4
frame_stride = 2
teacher-forcing transitions = 3 cặp
num_tokens = 576 hoặc layout tùy encoder/cache cụ thể
latent_dim thường khác NN-JEPA nếu dùng ViT-L 256/D=1024
```

Khác biệt quan trọng:

```text
NN-JEPA lấy 8 frame thật trong một sample, nên loss trung bình qua nhiều transition hơn
JEPA dùng horizon ngắn hơn, nhẹ hơn, train/eval ổn định RAM/VRAM hơn
cùng val session nhưng số window hợp lệ và số transition tính loss khác nhau
encoder/feature dim khác thì L1 latent tuyệt đối cũng không so sánh trực tiếp được
```

Vì vậy nếu muốn so sánh hai code chính xác hơn, không nên chỉ nhìn `val_loss`.

## Metric nên dùng khi so sánh hai bên

Điều kiện so sánh công bằng tối thiểu:

```text
cùng held-out val sessions
cùng frame_stride hoặc cùng effective fps
cùng horizon đánh giá
cùng số start frame hoặc cùng danh sách eval window
không dùng test alias val để tuyên bố kết quả cuối
```

Nếu feature encoder khác nhau, không nên so trực tiếp `rollout_l1_hk` tuyệt đối, vì scale latent có thể khác. Khi đó nên ưu tiên ratio:

```text
primary metric: ratio_h1, ratio_h2, ratio_h3
secondary metric: rollout_l1_h1/h2/h3
support metric: identity_l1_h1/h2/h3
training metric: val/loss, val/teacher_forcing_loss, val/rollout_loss
planning metric: planned_final_l1 / groundtruth_final_l1 / zero_action_final_l1
```

`ratio_hk` đáng tin hơn để so giữa hai hệ nếu latent scale khác nhau, vì nó chuẩn hóa lỗi model theo identity baseline trong cùng latent space của chính hệ đó.

## Cách code hiện tại chạy val mỗi epoch

Trong `src/tools/train_rc_jepa_ac_features.py`, mỗi epoch chạy theo thứ tự:

```text
1. train epoch
2. save checkpoint phase train_complete_waiting_val
3. chạy val thường bằng run_epoch
4. nếu val_rollout_eval_horizon > 0 thì chạy final_rollout_identity_eval trên val
5. merge metric rollout-vs-identity vào val_metrics
6. ghi history.json
7. in summary terminal
8. lưu last.pt
9. nếu val/loss tốt hơn thì lưu best.pt
10. log W&B train/*, val/*, best/* theo global_step
```

Tên hàm `final_rollout_identity_eval` được giữ lại vì ban đầu nó chỉ dùng cho final eval. Hiện hàm này được tái sử dụng cho cả per-epoch val eval bằng tham số `label`.

Tham số mới:

```text
--val-rollout-eval-horizon
--val-rollout-eval-max-batches
```

Ý nghĩa:

```text
val_rollout_eval_horizon = 0: tắt per-epoch rollout-vs-identity eval
val_rollout_eval_horizon = 3: đo h1, h2, h3 sau mỗi epoch
val_rollout_eval_max_batches = 256: chỉ sample tối đa 256 val batches để đỡ chậm
val_rollout_eval_max_batches = 0: chạy full val split
```

Với `eval_batch_size=1` và `max_batches=256`, mỗi epoch sẽ đo thêm tối đa 256 sample val cho rollout-vs-identity. Đây là đánh đổi hợp lý ban đầu vì full val hiện có khoảng 20,501 window theo frame_stride2, chạy full sẽ lâu hơn đáng kể.

## Dòng terminal sau mỗi epoch sẽ có gì

Sau thay đổi hiện tại, nếu `val_rollout_eval_horizon=3`, dòng cuối epoch sẽ có dạng:

```text
[epoch 001] train_loss=... val_loss=... train_tf=... val_tf=... train_rollout=... val_rollout=... val_rollout_l1_h1=... val_identity_l1_h1=... val_ratio_h1=... val_rollout_l1_h2=... val_identity_l1_h2=... val_ratio_h2=... val_rollout_l1_h3=... val_identity_l1_h3=... val_ratio_h3=...
```

Nếu horizon bị tắt, dòng terminal chỉ in loss cơ bản.

## W&B sẽ hiện những gì

Các metric theo epoch:

```text
train/loss
train/teacher_forcing_loss
train/rollout_loss
val/loss
val/teacher_forcing_loss
val/rollout_loss
val/rollout_l1_h1
val/identity_l1_h1
val/ratio_h1
val/rollout_l1_h2
val/identity_l1_h2
val/ratio_h2
val/rollout_l1_h3
val/identity_l1_h3
val/ratio_h3
best/val_loss
lr
```

Các metric cuối train sau khi load `best.pt`:

```text
final_eval_val/rollout_l1_h1
final_eval_val/identity_l1_h1
final_eval_val/ratio_h1
final_eval_val/rollout_l1_h2
final_eval_val/identity_l1_h2
final_eval_val/ratio_h2
final_eval_val/rollout_l1_h3
final_eval_val/identity_l1_h3
final_eval_val/ratio_h3
```

Các metric batch train vẫn giữ nguyên:

```text
train_batch/loss
train_batch/teacher_forcing_loss
train_batch/rollout_loss
train_batch/grad_pre_clip/*
train_batch/grad_post_clip/*
train_batch/param/*
```

## Offline planning metric có cần chạy xe thật không?

Không cần chạy xe thật. Đây là offline planning eval trên dữ liệu đã thu sẵn.

Ý tưởng:

```text
1. Lấy một sample trong val.
2. Lấy latent frame đầu làm trạng thái hiện tại: z_t.
3. Lấy latent frame tương lai trong cùng sample làm goal: z_{t+k}.
4. Cho CEM planner thử nhiều chuỗi action giả lập.
5. Với mỗi chuỗi action, dùng predictor/world model rollout ra z_hat_{t+k}.
6. Chấm điểm bằng khoảng cách L1 giữa z_hat_{t+k} và z_{t+k}.
```

Vì `z_{t+k}` đã có trong log cũ nên không cần chạy xe ngoài đời. World model đóng vai trò simulator latent. CEM planner chỉ tìm action mà model tin rằng sẽ đưa latent cuối tới gần goal.

Điểm phải nhớ: metric này không chứng minh xe thật sẽ chạy đúng ngoài đời. Nó chỉ kiểm tra:

```text
nếu tin predictor là simulator,
planner có tìm được action tốt hơn baseline không?
```

Các metric planning:

```text
mean_planned_final_l1
  L1 giữa latent cuối do action CEM tạo ra và goal latent.
  Càng thấp càng tốt.

mean_groundtruth_final_l1
  L1 giữa latent cuối khi dùng action thật trong log và goal latent.
  Đây là baseline "teacher/ground-truth action" nhưng vẫn rollout qua predictor.

mean_zero_action_final_l1
  L1 giữa latent cuối khi dùng toàn action 0 và goal latent.
  Đây là baseline đứng yên/không điều khiển.

planned_zero_ratio
  mean_planned_final_l1 / mean_zero_action_final_l1.
  < 1 nghĩa là planner tốt hơn zero-action baseline.

planned_groundtruth_ratio
  mean_planned_final_l1 / mean_groundtruth_final_l1.
  < 1 nghĩa là theo predictor, action CEM còn đưa latent tới goal tốt hơn action thật trong log.
  Không nên hiểu chỉ số này là CEM chắc chắn lái thật tốt hơn người/data.

mean_first_action_mae
  Sai khác trung bình giữa action đầu tiên CEM chọn và action thật trong log.
  Đây chỉ là metric phụ vì một goal có thể đạt bằng nhiều chuỗi action khác nhau.
```

Metric quan trọng nhất để xem planner có ích không:

```text
planned_zero_ratio < 1
```

Metric nên xem tiếp:

```text
mean_planned_final_l1
planned_groundtruth_ratio
mean_first_action_mae
```

## Planning metric đã nằm trong final eval chưa?

Đã thêm vào final eval dưới dạng optional.

Hiện tại train script có hai nhóm final eval:

```text
final_eval_val/*
  rollout-vs-identity metric của world model.

final_planning_val/*
  offline CEM planning metric trên val.
```

Khác biệt:

```text
final_eval_val dùng action thật/rollout model để xem dynamics có tốt hơn identity baseline không.
final_planning_val dùng CEM để tự tìm action rồi so với zero-action và ground-truth action.
```

Vì CEM planning tốn thời gian hơn nhiều so với val loss, nó không nên chạy sau mỗi epoch. Nó chỉ chạy cuối train sau khi đã load lại `best.pt`.

Các tham số mới:

```text
final_planning_eval_samples
  Số sample val dùng cho final planning eval.
  0 nghĩa là tắt.

final_planning_horizon
  Số bước action CEM rollout.
  0 nghĩa là dùng auto_steps.

final_planning_goal_offset
  Goal frame offset trong sample.
  0 nghĩa là bằng final_planning_horizon.

final_planning_cem_samples
  Số action sequence candidate mỗi vòng CEM.

final_planning_cem_elites
  Số candidate tốt nhất dùng để update phân phối CEM.

final_planning_cem_iters
  Số vòng refine CEM.
```

Config official-lite mixed hiện đã bật final planning eval nhỏ:

```yaml
# base mixed
final_planning_eval_samples: 16
final_planning_cem_samples: 32
final_planning_cem_elites: 8
final_planning_cem_iters: 3

# tiny mixed
final_planning_eval_samples: 32
final_planning_cem_samples: 64
final_planning_cem_elites: 8
final_planning_cem_iters: 3
```

Output cuối train:

```text
checkpoints/.../final_eval_val.json
checkpoints/.../final_planning_val.json
checkpoints/.../final_metrics.json
```

W&B metric cuối train:

```text
final_planning_val/mean_planned_final_l1
final_planning_val/mean_groundtruth_final_l1
final_planning_val/mean_zero_action_final_l1
final_planning_val/planned_zero_ratio
final_planning_val/planned_groundtruth_ratio
final_planning_val/mean_first_action_mae
final_planning_val/mean_first_action_mae/steering_cmd_t
final_planning_val/mean_first_action_mae/throttle_cmd_t
```

## Khi nào nên chạy tool planning riêng?

Final planning eval trong train chỉ là smoke/summary cuối train. Nó dùng ít sample để không làm train qua đêm bị kéo dài quá nhiều.

Vẫn nên chạy tool riêng khi cần:

```text
1. tăng max_samples lên vài trăm hoặc full val
2. thử nhiều horizon khác nhau
3. thử CEM samples/iters lớn hơn
4. xuất JSONL/CSV per-record
5. vẽ SVG so sánh planned/groundtruth/zero
```

Tool riêng:

```bash
PYTHONPATH=src python3 -m tools.plan_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features_servo_old_mix_official_lite_base_frame_stride2/best.pt \
  --features-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --split val \
  --max-samples 128 \
  --horizon 2 \
  --cem-samples 128 \
  --cem-elites 16 \
  --cem-iters 4
```

Vẽ đồ thị:

```bash
PYTHONPATH=src python3 -m tools.plot_rc_jepa_planning \
  --planning-jsonl checkpoints/rc_jepa_ac_vitb_features_servo_old_mix_official_lite_base_frame_stride2/planning/planning_val.jsonl
```

## Lệnh train official-lite base mixed

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2
```

Lệnh này dùng:

```text
model.type = official_lite
model.size = base
predictor params ~= 19,707,648
batch_size = 2
eval_batch_size = 1
amp_dtype = bf16
val_rollout_eval_horizon = 3
val_rollout_eval_max_batches = 256
final_planning_eval_samples = 16
final_planning_cem_samples = 32
final_planning_cem_iters = 3
skip_test = true
```

## Lệnh train official-lite tiny mixed

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny_mix_oldservo_frame_stride2
```

Lệnh này dùng:

```text
model.type = official_lite
model.size = tiny
predictor params ~= 595,456
batch_size = 32
eval_batch_size = 2
amp_dtype = bf16
val_rollout_eval_horizon = 3
val_rollout_eval_max_batches = 256
final_planning_eval_samples = 32
final_planning_cem_samples = 64
final_planning_cem_iters = 3
skip_test = true
```

## Override nếu val rollout eval làm train chậm

Tắt metric rollout-vs-identity per epoch:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2 \
  train.val_rollout_eval_horizon=0
```

Chạy full val rollout eval, không giới hạn batch:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2 \
  train.val_rollout_eval_max_batches=0
```

Khuyến nghị hiện tại: giữ `val_rollout_eval_max_batches=256` trong lúc thử nghiệm để không kéo dài epoch quá nhiều. Khi đã chọn vài checkpoint tốt, chạy eval offline/full val riêng để báo cáo kết quả.

Tắt final planning eval nếu muốn train kết thúc nhanh hơn:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2 \
  train.final_planning_eval_samples=0
```

Tăng số sample planning cuối train:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2 \
  train.final_planning_eval_samples=64
```

## Kết luận thực dụng

Để theo dõi train qua đêm:

```text
1. nhìn val/loss để biết checkpoint có cải thiện không
2. nhìn val/ratio_h1, val/ratio_h2, val/ratio_h3 để biết model có hơn identity baseline không
3. ratio < 1 là tín hiệu tốt
4. ratio càng thấp càng tốt, nhưng phải xem cùng identity_l1 để biết val split có chuyển động đủ không
5. không dùng test alias val làm kết luận cuối cùng
```

Nếu mục tiêu là chọn model để đưa sang planner/inference, ưu tiên checkpoint có:

```text
val/loss thấp
val/rollout_loss thấp
val/ratio_h1/h2/h3 ổn định dưới 1
final_eval_val/ratio_h1/h2/h3 dưới 1 khi chạy full hoặc sample đủ lớn
final_planning_val/planned_zero_ratio dưới 1
planner offline riêng có planned_final_l1 tốt hơn zero_action_final_l1 trên nhiều sample hơn
```
