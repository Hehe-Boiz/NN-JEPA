# HANDOFF - NN-JEPA RC Car JEPA-AC

## Cập nhật bàn giao - 2026-06-10

Repo hiện tại là `NN-JEPA`, đây là repo train. Repo `JEPA/` chỉ dùng cho phần cứng/recorder/sync/staging. Không sửa code gốc trong `vjepa2/`.

Tài liệu audit/báo cáo gần nhất:

```text
doc/full_audit_report_20260608.md
doc/bao_cao_vjepa2_ac_vs_nn_jepa.md
doc/bao_cao_kha_nang_dung_predictor_ac_official.md
doc/bao_cao_trien_khai_official_lite_predictor.md
doc/bao_cao_oom_eval_val_vs_train_batch.md
doc/bao_cao_jepa_update_20260609_inference.md
doc/ke_hoach_inference_an_toan_nn_jepa.md
doc/bao_cao_trien_khai_experiment_servo_cu.md
doc/bao_cao_metric_danh_gia_val_rollout_identity.md
```

Kết quả audit mới nhất:

```text
compileall: pass
unit tests: 47/47 pass
git diff --check: pass
smoke train-val-test-checkpoint: pass, loss hữu hạn
smoke offline CEM planner: pass với max-samples=1, horizon=1, cem-samples=2
smoke planning plot SVG: pass trên /tmp/nn_jepa_plan_smoke/planning_test.jsonl
Hydra dry-run rc_jepa_tiny_newdata: pass
Hydra wandb.continue_run=true/false: pass
feature cache ViT-B fp32: đã xóa để giải phóng disk
feature cache baseline ViT-B fp16: chưa extract lại
feature cache servo_old_mix_v1 ViT-B fp16: đã extract xong, 211 session, 212840 frame
official_lite predictor trước đó: pass shape/mask/RoPE/loss smoke
action-block causal mask: khớp source vjepa2 cho case thật [4624, 4624]
RoPE rotation: khớp source vjepa2, max diff = 0.0
per-epoch val rollout-vs-identity metrics: đã thêm, py_compile pass, Hydra dry-run base mixed pass
final planning eval: đã thêm `final_planning_val/*` optional cuối train trên best.pt
feature train sampler: giữ `global` mặc định, official-lite mixed dùng `train_sampler=session`, `eval_sampler=session`
git diff --check: pass
```

Blocker môi trường hiện tại trước khi train thật:

```text
GPU/CUDA: nvidia-smi fail, chưa giao tiếp được NVIDIA driver
```

Việc cần làm ngay trước train baseline: sửa GPU/driver rồi extract lại feature fp16. Riêng `servo_old_mix_v1` đã có feature cache fp16 hợp lệ và có thể train ngay bằng config mixed.

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
data/raw: 181 session
train: 103208 samples, 82623 windows
val:   57829 samples, 48103 windows
test:  57829 samples, 48103 windows  # alias của val
feature cache path: data/processed/features/vjepa2_1_vitb_384_ema_fp16
feature cache status: chưa extract lại sau khi xóa fp32
expected metadata: 161037 frame, 576 token/frame, embed_dim 768, fp16
mixed feature cache path: data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
mixed feature cache status: đã extract xong, 211 session, 212840 frame, 576 token/frame, embed_dim 768, fp16, source_frame_path
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

NN-JEPA đã có planner offline kiểu JEPA:

```text
src/tools/rc_jepa_ac_cem_planner.py
src/tools/plan_rc_jepa_ac_features.py
src/tools/plot_rc_jepa_planning.py
```

Planner này load checkpoint predictor + feature cache, lấy context latent và goal
latent trong sample, chạy CEM trên action raw, normalize action đúng như lúc train,
rollout predictor rồi ghi JSONL/CSV. Đây là bước offline inference/eval, chưa gửi
lệnh xuống xe và chưa thay thế live closed-loop.

Tool plot đọc `planning_*.jsonl` và xuất SVG dependency-free:

```text
latent_l1_comparison.svg
first_action_planned_vs_groundtruth.svg
first_action_abs_error.svg
action_sequence_record_*.svg
```

Drive/staging đã audit trước đó:

```text
JEPA/data/drive_zips:
  65 zip khớp Drive, 0 differences
  63 zip top-level session_20260607_*.zip đã từng được dùng train trước khi có thêm data mới
  2 zip cũ trong trong nhà/ staging-only, không train

JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV:
  30 session cũ 20260605
  56027 file
  actions_synced.csv / imu_synced.csv đủ 30/30
  không đưa vào data/raw baseline
```

Data servo cũ đã được khôi phục thủ công từ 3 zip:

```text
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-001.zip
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-002.zip
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV-20260608T112340Z-3-003.zip
```

Sau giải nén, session `session_20260605_155710` thiếu synced; đã chạy lại `jepa_wm.data.sync`, kết quả giữ `47`, bỏ `1`, `offset=100ms +imu`.

Web viewer hiện có 3 job riêng:

```text
Sync Drive: kéo zip mới, extract top-level session, chạy sensor sync; không preprocess
Preprocess: chạy tools.preprocess_data
Extract V-JEPA Features: chạy tools.extract_vjepa_features
```

Web viewer có thanh tiến trình trong panel `Data Ops`:

- `Sync Drive`: progress theo stage; byte-level/tốc độ tải của `rclone -P` nằm trong `Job Log`.
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
  --dtype fp16
```

Nếu đổi preset encoder thì phải train predictor lại với đúng feature dir mới.

Chạy web:

```bash
conda activate nn-jepa
PYTHONPATH=src python3 -m tools.session_web_viewer
```

Mở:

```text
http://127.0.0.1:8765
```

Feature extractor đã có fast-skip:

```text
nếu session cache đúng shape/dtype -> skip session
nếu toàn bộ cache đầy đủ + metadata khớp -> kết thúc sớm, không load encoder/checkpoint
cache baseline ViT-B fp16 cần extract lại
mixed servo cũ có --seed-from-features-dir để reuse feature baseline khi metadata encoder khớp
```

Experiment servo cũ đã triển khai riêng:

```text
tool build: src/tools/build_servo_experiment_dataset.py
mixed root: data/experiments/servo_old_mix_v1
mixed hydra: configs/hydra/experiment/rc_jepa_tiny_mix_oldservo_frame_stride2.yaml
```

Audit experiment servo cũ:

```text
mixed meaning: current_servo trong data/raw + old_servo ngoài JEPA/data/drive_extra_nonzip
mixed split file: data/split_vjepa_ac_car.json
mixed sessions: 181 current_servo + 30 old_servo = 211
mixed samples train/val/test: 182558 / 30282 / 30282 alias val
mixed frame_stride=2 windows train/val/test: 128566 / 20501 / 20501 alias val
mixed feature cache fp16: đã extract lại sạch từ raw frame gốc, 211 session, 212840 frame, dtype fp16, image_path_key=source_frame_path
mixed DataLoader frame_stride=2 windows: train 128566, val 20501
mixed sample shape: latents (4608,768), states (8,5), actions (7,2)
old-only: đã xóa, không còn Hydra config riêng
preprocess_report bad_sessions: 0
feature extraction mixed hiện tại đã chạy lại bằng extractor mới; train guard metadata pass vì cache có image_path_key=source_frame_path
```

Default quan trọng:

```text
feature cache: data/processed/features/vjepa2_1_vitb_384_ema_fp16
feature train output: checkpoints/rc_jepa_ac_vitb_features_20260607
eval/infer default checkpoint: checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt
V-JEPA checkpoint: checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
encoder: vit_base_384
checkpoint key: ema_encoder
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

`simple base` vẫn là mặc định. `tiny` dùng `--model-size tiny` để thử pipeline nhanh. `official_lite tiny` nên chạy batch nhỏ hơn, ví dụ `batch_size=4`, `eval_batch_size=1`.

Kiểm tra gần nhất:

```text
compileall: pass
unit tests: 47/47 pass
Hydra official_lite dry-run: pass
official_lite full-token smoke loss: pass
web smoke test: /, /app.js, /styles.css, /api/sessions, /api/jobs đều HTTP 200
rclone/session_web_viewer process nền: không còn
```

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
data/processed/manifests/train.jsonl
data/processed/manifests/val.jsonl
data/processed/manifests/test.jsonl
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
predictor chọn bằng predictor_type:
  simple -> SimpleACPredictor
  official_lite -> VJepaStyleACPredictor
```

Encoder:

- Load từ local `vjepa2/`.
- Dùng checkpoint V-JEPA 2.1.
- Checkpoint key mặc định: `ema_encoder`.
- Freeze toàn bộ parameter.
- Luôn `eval()`.
- Chạy `torch.no_grad()`.
- Chỉ dùng để tạo latent target.

Predictor `simple`:

- Causal transformer nhỏ.
- Nhận latent tokens.
- Nhận action token.
- Nhận state token.
- Là phần duy nhất được train.

Predictor `official_lite`:

- Bám source V-JEPA AC hơn `simple`.
- Token layout là `[action, state, patch tokens]`.
- Dùng action-block causal attention mask giống source `vjepa2`.
- Dùng RoPE attention frame/height/width.
- Output vẫn là latent patch tokens.
- Không phải exact official reproduction vì state/action RC là 5D/2D, còn DROID official là 7D/7D.

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
  --manifest-dir data/processed/manifests \
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

Lệnh train khuyến nghị hiện tại là train predictor từ feature cache:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607 \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --num-workers 8 \
  --lr 1e-4 \
  --warmup-epochs 4 \
  --warmup-start-factor 0.1 \
  --min-lr-ratio 0.1 \
  --early-stopping-patience 15 \
  --wandb-project nn-jepa-rc
```

Lệnh train `tiny` để thử nhanh:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --model-size tiny \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_tiny \
  --epochs 20 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --num-workers 8 \
  --lr 1e-4 \
  --warmup-epochs 2 \
  --early-stopping-patience 5 \
  --wandb-project nn-jepa-rc \
  --wandb-tags tiny
```

Lệnh train `official_lite tiny` để thử predictor gần source V-JEPA AC hơn:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --predictor-type official_lite \
  --model-size tiny \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny \
  --epochs 20 \
  --batch-size 4 \
  --eval-batch-size 1 \
  --num-workers 8 \
  --lr 1e-4 \
  --warmup-epochs 2 \
  --early-stopping-patience 5 \
  --wandb-project nn-jepa-rc \
  --wandb-tags official_lite tiny
```

Hydra equivalent:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny
```

## Trạng thái kiểm tra

Đã chạy gần nhất:

```bash
python -m compileall src tests
PYTHONPATH=src python -m unittest discover -s tests -v
```

Kết quả: `47/47` pass. Torch đang dùng trực tiếp, không có fallback import.

## Cập nhật mới nhất cần nhớ

Các lỗi logic train đã được sửa:

- `rc_jepa_ac` rollout không dùng state tương lai thật nữa.
- Sequence dataset chặn window bị đứt frame/time bằng `AC_MAX_FRAME_INDEX_GAP` và `AC_MAX_TIME_GAP_SEC`.
- State/action numeric input được normalize bằng stats từ train manifest.
- Checkpoint train lưu metadata normalization để inference dùng lại đúng scale.
- Outlier robust đang bật bằng `REMOVE_SIMPLE_OUTLIERS = True`.

Manifest hiện tại sau preprocess:

```text
train: 103208 samples, 82623 windows
val:   57829 samples, 48103 windows
test:  57829 samples, 48103 windows  # alias của val
```

Không còn test split độc lập. Nếu gặp manifest cũ, dùng `src/tools/drop_test_split.py` để nhập test cũ vào val và ghi lại `test.jsonl` alias val. Tool này đã idempotent, chạy lại sẽ không nhân đôi val.

Report:

```text
data/processed/reports/preprocess_report.json
```

W&B đang bật mặc định cho hai script train:

```text
project: nn-jepa-rc
metrics: train/*, val/*, best/val_loss, lr
```

`test/*` chỉ có nếu chạy eval/test riêng hoặc bật `--run-test`; train feature-cache mặc định bỏ test và chỉ dùng val.

Tắt bằng:

```bash
--no-wandb
```

## Việc cần làm tiếp

1. Sửa GPU/driver trước, vì `nvidia-smi` đang fail.
2. Baseline feature cache ViT-B fp16 cần extract lại vì fp32 đã xóa để giảm dung lượng.
3. Nếu Drive có data mới: mở web và bấm `Sync Drive`, theo dõi progress/log.
4. Kiểm tra session mới trong viewer.
5. Bấm `Preprocess` để rebuild manifest, theo dõi progress theo session.
6. Chọn `Feature model`, rồi bấm `Extract V-JEPA Features`; cache đúng thì skip, session mới thì extract.
7. Với experiment servo cũ: `servo_old_mix_v1` đã có feature cache fp16 hợp lệ, có thể train.
8. Train thử `simple tiny` mixed servo bằng Hydra: `experiment=rc_jepa_tiny_mix_oldservo_frame_stride2`.
9. Train `official_lite base` mixed servo mạnh nhất hiện tại bằng Hydra: `experiment=rc_jepa_official_lite_base_mix_oldservo_frame_stride2`.
10. Official-lite mixed dùng session sampler: 1 batch cùng session, shuffle session/window trong train, drain session rồi sang session kế tiếp.
11. Nếu base OOM/chậm quá, quay về `official_lite tiny`: `experiment=rc_jepa_official_lite_tiny_mix_oldservo_frame_stride2`.
12. Theo dõi `val/rollout_l1_h*`, `val/identity_l1_h*`, `val/ratio_h*`; `ratio_h* < 1` nghĩa là model tốt hơn identity baseline.
13. Cuối train official-lite mixed sẽ có `final_planning_val/*`; `planned_zero_ratio < 1` nghĩa là CEM planner tốt hơn zero-action baseline theo world model.
14. Eval/test bằng `tools.eval_rc_jepa_ac_features`.
15. Sau khi world model loss ổn, chạy planning offline riêng với nhiều sample hơn hoặc thêm planner/MPC/policy head.

## Ràng buộc cần nhớ

- Không đụng `vjepa2/`.
- Giữ code đơn giản, dễ đọc, dễ sửa.
- Không thêm fallback import cho torch.
- Không đổi baseline behavior cloning nếu user không yêu cầu.
- Ưu tiên đặt biến dễ chỉnh trong `settings.py`.
- Tránh dùng từ `context` trong code RC mới để không nhầm với context token của JEPA.
