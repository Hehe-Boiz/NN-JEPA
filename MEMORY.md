# MEMORY - Dự án NN-JEPA cho xe RC

## Cập nhật mới nhất - 2026-06-09

Repo hiện tại là `NN-JEPA`, dùng cho phần train. Repo `JEPA/` chỉ phục vụ phần cứng, recorder, sync sensor và staging data.

Các báo cáo quan trọng gần nhất:

```text
doc/full_audit_report_20260608.md
doc/bao_cao_vjepa2_ac_vs_nn_jepa.md
doc/bao_cao_kha_nang_dung_predictor_ac_official.md
doc/bao_cao_trien_khai_official_lite_predictor.md
doc/bao_cao_oom_eval_val_vs_train_batch.md
doc/bao_cao_jepa_update_20260609_inference.md
doc/ke_hoach_inference_an_toan_nn_jepa.md
```

Kết quả audit mới nhất:

```text
compileall: pass
unit tests: 46/46 pass
git diff --check: pass
smoke train-val-test-checkpoint: pass, loss hữu hạn
smoke offline CEM planner: pass với max-samples=1, horizon=1, cem-samples=2
smoke planning plot SVG: pass trên /tmp/nn_jepa_plan_smoke/planning_test.jsonl
Hydra dry-run rc_jepa_tiny_newdata: pass
Hydra wandb.continue_run=true/false: pass
feature cache ViT-B fp32: manifest 100 session, 100 json, 100 npy, missing = 0
official_lite predictor trước đó: pass shape/mask/RoPE/loss smoke
action-block causal mask: khớp source vjepa2 cho case thật [4624, 4624]
RoPE rotation: khớp source vjepa2, max diff = 0.0
git diff --check: pass
```

Blocker môi trường hiện tại trước khi train thật:

```text
GPU/CUDA: nvidia-smi fail, chưa giao tiếp được NVIDIA driver
```

Vì vậy cần sửa GPU/driver trước khi train thật. Feature cache hiện đã đủ theo manifest mới nhất.

`JEPA/` vừa kiểm tra sau khi pull:

```text
HEAD: e841729 Update HANDOFF: overnight results + inference blockers
Commit kỹ thuật chính ngay trước đó: 30215d8 Add VJEPA2ACCar
Thay đổi chính: VJEPA2ACCar patch-token V-JEPA-2-AC cho xe RC, full IMU 10D,
encode_patch 256px ViT-L, ACClipDataset memmap, CEMPlannerAC, CarDynamics,
train_ac_car và báo cáo inference blocker.
Kết quả JEPA báo cáo: VJEPA2ACCar rollout@1 ratio 0.826, rollout@3 ratio 0.775,
tốt hơn pooled baseline rollout@1 ratio 0.867.
Ảnh hưởng tới NN-JEPA: chưa ảnh hưởng trực tiếp; nên port theo từng bước, ưu tiên
offline CEM/eval_goal_reaching trước closed-loop thật.
```

Trạng thái data/manifest hiện tại:

```text
data/raw: 100 session
data/raw session cũ 20260605: không còn trong data/raw theo audit trước
processed manifest:
  train: 84766 samples, 68139 windows
  val:   9470 samples, 8259 windows
  test:  20310 samples, 16138 windows
feature cache:
  path: data/processed/features/vjepa2_1_vitb_384_ema_fp32
  files: 100 .npy + 100 .json
  metadata: 114546 frame, 576 token/frame, embed_dim 768, fp32
  missing_json_count: 0
  missing_npy_count: 0
```

Ghi chú inference sau khi đọc JEPA mới:

```text
JEPA chưa có scripts/inference_loop.py hoàn chỉnh.
pc_stream_view.py hiện chỉ nhận phone -> PC, chưa gửi action ngược.
robot/capture/controller.py còn UDP cũ; JEPA khuyến nghị đổi sang dongle serial ESP-NOW.
Hướng đúng cho NN-JEPA: offline CEM planner/eval trước, sau đó live dry-run, cuối cùng mới closed-loop thật.
Không dùng lẫn checkpoint JEPA VJEPA2ACCar với NN-JEPA hiện tại vì feature format khác:
  JEPA: ViT-L 256, 256 token/frame, D=1024
  NN-JEPA: ViT-B 384, 576 token/frame, D=768
```

NN-JEPA đã có bước inference offline kiểu planner:

```text
src/tools/rc_jepa_ac_cem_planner.py
src/tools/plan_rc_jepa_ac_features.py
src/tools/plot_rc_jepa_planning.py
```

Ý nghĩa: load checkpoint predictor + feature cache, lấy `z_t` và goal `z_{t+k}`,
chạy CEM trên action raw, tự normalize action đúng như train, rollout predictor,
chọn action sequence có latent cuối gần goal nhất. Đây chưa phải closed-loop thật
và chưa gửi lệnh xuống xe.

Tool plot đọc `planning_*.jsonl` và xuất SVG dependency-free:

```text
latent_l1_comparison.svg
first_action_planned_vs_groundtruth.svg
first_action_abs_error.svg
action_sequence_record_*.svg
```

Drive/staging đã audit trước đó:

```text
JEPA/data/drive_zips: 65 zip khớp Drive, 0 differences
  63 zip top-level session_20260607_*.zip đã từng được dùng cho train trước khi có thêm data mới
  2 zip cũ trong trong nhà/ chỉ staging-only, không train
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV:
  28 session cũ 20260605
  53403 file
  actions_synced.csv / imu_synced.csv đủ 28/28
  không nằm trong data/raw, không dùng train hiện tại
```

`data servo cũ KDS 680HV` đã được khôi phục thủ công từ 3 zip:

```text
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-001.zip
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-002.zip
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-003.zip
```

Session `session_20260605_155710` trong data servo cũ từng thiếu synced; đã chạy lại:

```bash
PYTHONPATH=JEPA/src python -m jepa_wm.data.sync \
  "JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV/session_20260605_155710"
```

Kết quả: giữ `47` frame, bỏ `1`, `offset=100ms +imu`.

Web viewer hiện có 3 job riêng:

```text
Sync Drive: rclone copy zip mới + extract top-level session + chạy sensor sync, không preprocess
Preprocess: chạy tools.preprocess_data
Extract V-JEPA Features: chạy tools.extract_vjepa_features
```

Web viewer có thanh tiến trình trong panel `Data Ops`:

- `Sync Drive`: progress theo stage; byte-level/tốc độ tải của `rclone -P` vẫn xem trong `Job Log`.
- `Preprocess`: progress theo số session đã xử lý.
- `Extract V-JEPA Features`: có dropdown `Feature model`, progress theo số session đã extract/skip; nếu cache đủ thì báo 100% ngay.

Feature extractor có preset V-JEPA 2.1:

```text
vitb_384 -> vit_base_384, checkpoint_key=ema_encoder, output vjepa2_1_vitb_384_ema_<dtype>
vitl_384 -> vit_large_384, checkpoint_key=ema_encoder, output vjepa2_1_vitl_384_ema_<dtype>
vitg_384 -> vit_giant_384, checkpoint_key=target_encoder, output vjepa2_1_vitg_384_target_<dtype>
vitG_384 -> vit_gigantic_384, checkpoint_key=target_encoder, output vjepa2_1_vitG_384_target_<dtype>
```

Lệnh extract nên dùng preset:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --batch-size 32 \
  --dtype fp32
```

Nếu đổi preset encoder thì phải train predictor lại với đúng feature dir mới.

Feature extractor hiện có fast-skip:

- Nếu session đã có `.npy + .json` đúng shape/dtype thì skip session đó.
- Nếu toàn bộ cache đầy đủ và metadata khớp thì kết thúc sớm, không load encoder/checkpoint.
- Hiện cache chưa đầy đủ vì thiếu 37 session theo manifest mới, cần chạy extract bù sau khi GPU hoạt động.

Các default quan trọng hiện tại:

```text
feature train output: checkpoints/rc_jepa_ac_vitb_features_20260607
eval/infer default checkpoint: checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt
feature cache path: data/processed/features/vjepa2_1_vitb_384_ema_fp32
V-JEPA checkpoint: checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
checkpoint key: ema_encoder
encoder: vit_base_384
```

Predictor hiện tại có 2 loại:

```text
simple: baseline cũ, mặc định
official_lite: predictor mới theo hướng V-JEPA AC official-lite
```

Chọn bằng:

```bash
--predictor-type simple
--predictor-type official_lite
```

Preset predictor hiện tại với ViT-B 384 feature:

```text
simple tiny:          predictor_dim=128, depth=2, heads=4, params=670,464
simple small:         predictor_dim=256, depth=4, heads=4, params=3,706,112
simple base:          predictor_dim=512, depth=6, heads=8, params=20,007,680
official_lite tiny:   predictor_dim=128, depth=2, heads=4, params=595,456
official_lite small:  predictor_dim=256, depth=4, heads=4, params=3,556,096
official_lite base:   predictor_dim=512, depth=6, heads=8, params=19,707,648
```

`simple base` vẫn là mặc định. `tiny` dùng `--model-size tiny` để thử pipeline nhanh. `official_lite tiny` nên dùng batch nhỏ hơn, ví dụ `batch_size=4`, `eval_batch_size=1`.

Kiểm tra gần nhất:

```text
compileall: pass
unit tests: 41/41 pass
Hydra official_lite dry-run: pass
official_lite full-token smoke loss: pass
web smoke test: /, /app.js, /styles.css, /api/sessions, /api/jobs đều HTTP 200
không còn rclone/session_web_viewer process nền
```

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
predictor chọn bằng predictor_type:
  simple -> SimpleACPredictor
  official_lite -> VJepaStyleACPredictor
```

Encoder:

- Import từ `vjepa2/app/vjepa_2_1/models/vision_transformer.py`.
- Load checkpoint V-JEPA 2.1.
- Checkpoint key mặc định là `ema_encoder`.
- Freeze toàn bộ parameter.
- Luôn ở `eval()`.
- Chạy trong `torch.no_grad()`.
- Chỉ tạo latent target tokens.

Predictor `simple`:

- Là causal transformer nhỏ, dễ đọc hơn predictor public.
- Nhận latent tokens.
- Nhận action token.
- Nhận state token.
- Chỉ predictor được train.

Predictor `official_lite`:

- Bám theo source V-JEPA AC hơn `simple`.
- Token layout là `[action, state, patch tokens]`.
- Dùng action-block causal attention mask giống source `vjepa2`.
- Dùng RoPE attention frame/height/width.
- Output vẫn là latent patch tokens.
- Không phải exact official reproduction vì state/action RC là 5D/2D, còn DROID official là 7D/7D.

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

Lưu ý hiện tại: môi trường `nn-jepa` có `torch`; test tensor/model đã chạy và pass. Code vẫn import torch trực tiếp, không có fallback import.

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

Preprocess hiện tại đã rebuild manifest với outlier robust:

```text
train: 84766 samples, 68139 windows
val:   9470 samples, 8259 windows
test:  20310 samples, 16138 windows
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

1. Sửa GPU/driver trước, vì `nvidia-smi` đang fail.
2. Chạy extract bù feature cache ViT-B fp32; hiện manifest có 100 session nhưng cache thiếu 37 `.json`.
3. Nếu Drive có data mới: mở web, bấm `Sync Drive` và theo dõi progress/log.
4. Kiểm tra session mới trong web viewer.
5. Bấm `Preprocess` để rebuild manifest và theo dõi progress theo session.
6. Chọn `Feature model`, rồi bấm `Extract V-JEPA Features`; cache cũ đúng thì sẽ skip, session mới thì mới extract.
7. Train thử predictor `simple tiny` bằng `tools.train_rc_jepa_ac_features --predictor-type simple --model-size tiny`.
8. Train thử predictor `official_lite tiny` bằng `tools.train_rc_jepa_ac_features --predictor-type official_lite --model-size tiny --batch-size 4 --eval-batch-size 1`.
9. Khi pipeline/loss/log ổn, so sánh `simple base` với `official_lite small/base`.
10. Eval/test bằng `tools.eval_rc_jepa_ac_features`.
11. Sau khi loss ổn, thêm planner/MPC hoặc policy head để chọn action.
