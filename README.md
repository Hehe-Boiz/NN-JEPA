# NN-JEPA

Repo này là phần **train** cho xe RC. Repo `JEPA/` chỉ phục vụ phần cứng, recorder, sync.

Luồng hiện tại:

```text
data/raw/session_xxx/... -> preprocess -> data/processed/... -> train / viewer
```

## Hiện trạng quan trọng - 2026-06-10

Predictor hiện có 2 loại:

```text
simple          baseline cũ, mặc định
official_lite   predictor mới theo hướng V-JEPA AC official-lite
```

Chọn bằng:

```bash
--predictor-type simple
--predictor-type official_lite
```

Audit `official_lite` đã kiểm tra:

- action-block causal mask trong NN-JEPA khớp source `vjepa2` cho case thật `8 frame, 24x24 patch, 2 cond token`, shape `[4624, 4624]`
- RoPE rotation khớp source `vjepa2`, sai số max `0.0`
- token layout đúng `[action, state, patch tokens]`
- output vẫn là latent patch tokens
- `teacher_forcing_loss + rollout_loss` chạy được với `tokens_per_frame=576`, `latent_dim=768`
- `compileall`: pass
- unit tests: `47/47` pass
- `git diff --check`: pass
- smoke planner offline CEM: pass
- Hydra dry-run `rc_jepa_official_lite_tiny`: pass

Blocker hiện tại trước khi train thật:

```text
GPU: nvidia-smi đang fail vì không giao tiếp được NVIDIA driver
feature cache ViT-B fp32: đã xóa để giải phóng disk
feature cache ViT-B fp16: cần extract lại vào data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

Vì vậy trước khi train thật cần sửa GPU/driver để `nvidia-smi` và `torch.cuda.is_available()` hoạt động, rồi extract lại feature fp16.

Inference offline kiểu JEPA đã có:

```text
src/tools/rc_jepa_ac_cem_planner.py
src/tools/plan_rc_jepa_ac_features.py
doc/ke_hoach_inference_an_toan_nn_jepa.md
```

## Cập nhật repo `JEPA/` vừa kiểm tra

`JEPA/` đang ở commit:

```text
e841729 Update HANDOFF: overnight results + inference blockers
```

Các thay đổi chính trong `JEPA/`:

- thêm `VJEPA2ACCar` patch-token cho xe RC: `src/jepa_wm/models/vjepa2_ac_car.py`
- thêm dataset/feature patch-token: `src/jepa_wm/data/ac_clip.py`, `scripts/encode_patch.py`
- thêm train AC car: `src/jepa_wm/engine/train_ac_car.py`
- thêm CEM planner AC và dynamics: `src/jepa_wm/planning/cem.py`, `src/jepa_wm/planning/dynamics.py`
- thêm phần visual navigation/topological graph: `src/jepa_wm/nav/graph.py`, `scripts/build_graph.py`, `scripts/eval_navigation.py`, `scripts/eval_goal_reaching.py`, `scripts/viz_route.py`
- thêm/cập nhật config train cho servo hiện tại: `configs/train/vjepa_ac_towerpro.yaml`, `configs/train/vjepa_ac_mixed.yaml`
- cập nhật `robot/tools/pull_drive.py` và một số docs/eval script

Điểm quan trọng: code train hiện tại của NN-JEPA không tự thay đổi theo `JEPA/`. Phần đã port sang NN-JEPA hiện là planner CEM offline trên feature cache, không phải closed-loop phần cứng.

## Cấu trúc dữ liệu

Raw data nằm trong:

```text
data/raw/
  session_20260607_111842/
    frames/
    actions_synced.csv
    imu_synced.csv
    actions.csv
    telemetry.csv
    accel.csv
    gyro.csv
    rotvec.csv
    gps.csv
    meta.json
```

Processed data nằm trong:

```text
data/processed/
  images/
  manifests/
    train.jsonl
    val.jsonl
    test.jsonl
  reports/
    preprocess_report.json
```

## Chạy nhanh

Trước khi chạy các lệnh bên dưới:

```bash
conda activate nn-jepa
```

### 0. Đồng bộ data từ Google Drive

Nếu Drive `gdrive:JEPA` có session mới, chạy:

```bash
PYTHONPATH=src python3 -m tools.sync_drive_data \
  --check-zips \
  --preprocess
```

Tool này làm theo thứ tự:

- `rclone copy gdrive:JEPA JEPA/data/drive_zips --include '*.zip'`: incremental, file nào đã có và không đổi thì bỏ qua, chỉ tải file mới/đổi.
- kiểm tra zip bằng `rclone check` nếu bật `--check-zips`.
- chỉ extract các zip top-level dạng `session_*.zip` vào `data/raw/session_*`.
- không đưa zip/session cũ trong thư mục con `trong nhà/` hoặc folder `data servo cũ KDS 680HV/` vào `data/raw`.
- chạy sensor sync để tạo `actions_synced.csv` và `imu_synced.csv` cho session mới hoặc session đang thiếu file synced.
- nếu bật `--preprocess`, chạy lại preprocess để cập nhật `data/processed`.

Nếu chỉ muốn tải zip về staging, chưa extract/preprocess:

```bash
PYTHONPATH=src python3 -m tools.sync_drive_data \
  --skip-extract \
  --skip-sensor-sync
```

Nếu muốn tải cả phần non-zip cũ từ Drive để lưu riêng, thêm:

```bash
--sync-extra-nonzip
```

Lưu ý: phần non-zip cũ có nhiều file nhỏ nên tải rất chậm và không dùng cho train hiện tại.

Nếu một zip session trên Drive bị sửa sau khi local đã extract, tool sẽ không tự xoá session cũ. Khi thật sự muốn thay local bằng zip mới, chạy thêm:

```bash
--overwrite-changed
```

### 1. Preprocess lại dữ liệu

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

Pipeline sẽ:

- đọc toàn bộ session trong `data/raw`
- resize ảnh
- ghi ảnh processed vào `data/processed/images`
- tạo manifest train/val; `test.jsonl` chỉ là alias của `val.jsonl` để tương thích tool eval/infer cũ
- ghi report vào `data/processed/reports/preprocess_report.json`

Nếu đang có manifest cũ đã từng chia test độc lập, chuyển sang chế độ mới bằng:

```bash
PYTHONPATH=src python3 -m tools.drop_test_split --manifest-dir data/processed/manifests
```

Tool này idempotent: chạy lại nhiều lần không nhân đôi val.

### 2. Mở web xem session như video

```bash
PYTHONPATH=src python3 -m tools.session_web_viewer
```

Mở trình duyệt:

```text
http://127.0.0.1:8765
```

Web viewer hỗ trợ:

- xem `raw` hoặc `processed`
- chọn session
- play / pause
- kéo thanh frame
- chỉnh FPS
- phím tắt `Space`, `Left`, `Right`
- bấm `Sync Drive` để kéo zip mới từ Drive, extract session mới và sync sensor
- bấm `Preprocess` để resize ảnh, tạo lại manifest train/val sau khi sync xong
- chọn `Feature model`, rồi bấm `Extract V-JEPA Features` để chạy feature extractor từ web
- xem thanh tiến trình job trong panel `Data Ops`
- xem log job trực tiếp trong panel `Job Log`

Lưu ý khi chạy job từ web:

- mỗi lần chỉ chạy 1 job để tránh 2 tiến trình cùng ghi vào `data/raw` hoặc feature cache
- nút sync dùng cùng logic với `tools.sync_drive_data --check-zips`
- nút preprocess dùng cùng logic với `tools.preprocess_data`
- nút extract feature dùng preset V-JEPA 2.1; mặc định là `vitb_384`, checkpoint `ema_encoder`, cache `fp16`
- thanh tiến trình `Sync Drive` là progress theo stage; tốc độ/byte-level của `rclone -P` vẫn nằm trong `Job Log`
- thanh tiến trình `Preprocess` chạy theo số session đã xử lý
- thanh tiến trình `Extract V-JEPA Features` chạy theo số session đã extract/skip
- nếu feature đã tồn tại đúng shape/dtype, extractor sẽ skip session đó, không ghi lại
- nếu toàn bộ feature cache đã đầy đủ và metadata khớp, extractor kết thúc sớm mà không load encoder/checkpoint
- muốn dừng job đang chạy thì bấm `Cancel Running Job`

Thứ tự dùng 3 chức năng web:

1. Bấm `Sync Drive` khi Google Drive có session mới. Nút này chỉ kéo zip mới, extract session mới vào `data/raw`, rồi tạo `actions_synced.csv`/`imu_synced.csv` nếu thiếu.
2. Bấm `Preprocess` sau khi sync xong. Nút này resize ảnh, làm sạch data, chia train/val và tạo lại manifest. `test.jsonl` được ghi bằng chính `val` để tool cũ vẫn chạy được, nhưng không còn test split độc lập.
3. Chọn `Feature model`, rồi bấm `Extract V-JEPA Features` sau khi preprocess xong. Nút này encode frame bằng V-JEPA 2.1 và lưu feature cache; session nào đã extract đúng rồi thì skip.

### 3. Export nhanh 1 session ra GIF

```bash
PYTHONPATH=src python3 -m tools.export_session_gif \
  --session-id session_20260607_111842
```

GIF sẽ được ghi vào:

```text
data/previews/<session_id>.gif
```

### 4. Train baseline behavior cloning

```bash
PYTHONPATH=src python3 -m tools.train_rc_car
```

Model baseline nhận:

- ảnh
- state

và dự đoán:

- `steering_cmd_t`
- `throttle_cmd_t`

### 5. Train world model kiểu JEPA-AC với encoder V-JEPA 2.1 freeze

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4 \
  --warmup-epochs 4 \
  --warmup-start-factor 0.1 \
  --min-lr-ratio 0.1 \
  --early-stopping-patience 15 \
  --wandb-log-every 20
```

Lưu ý:

- file trong `vjepa2/` không bị sửa
- encoder V-JEPA 2.1 được load rồi freeze
- chỉ train predictor / world model
- checkpoint nhỏ nhất public nên dùng trước là ViT-B/16 80M: `vjepa2_1_vitb_dist_vitG_384.pt`
- LR mặc định hiện tại: `1e-4`
- lịch LR hiện tại: PyTorch `torch.optim.lr_scheduler.LambdaLR` với `linear warmup 4 epoch` rồi `cosine decay`
- early stopping mặc định: `15` epoch không cải thiện `val/loss`
- epoch warmup không tính vào patience của early stopping
- sau mỗi epoch sẽ lưu `last.pt`, `best.pt` và `epochs/epoch_xxx.pt`

Tham số V-JEPA 2.1 và NN-JEPA hiện tại:

```text
Nguồn chính:
- NN-JEPA encoder wrapper: src/models/rc_jepa_ac.py
- NN-JEPA train defaults: src/tools/train_rc_jepa_ac.py
- NN-JEPA global data defaults: src/data/settings.py
- V-JEPA 2.1 builders: vjepa2/app/vjepa_2_1/models/vision_transformer.py
- V-JEPA public robot config: vjepa2/configs/train/vitg16/droid-256px-8f.yaml
```

Encoder đang dùng mặc định trong NN-JEPA:

```text
encoder_name = vit_base_384
checkpoint_key = ema_encoder
checkpoint gợi ý = checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt
image_size = 384
patch_size = 16
tubelet_size = 2
embed_dim = 768
depth = 12
num_heads = 12
encoder_params = 86,833,152
```

Các encoder V-JEPA 2.1 mà NN-JEPA hiện expose qua CLI:

```text
name                params       embed_dim  depth  heads  image  patch  tubelet  tokens/frame
vit_small_384       22,182,912   384        12     6      384    16     2        576
vit_base_384        86,833,152   768        12     12     384    16     2        576
vit_large_384       304,680,960  1024       24     16     384    16     2        576
vit_giant_384       rất lớn      1408       40     22     384    16     2        576
vit_gigantic_384    rất lớn      1664       48     26     384    16     2        576
```

`vit_giant_384` và `vit_gigantic_384` dùng builder xformers trong source `vjepa2`; không khuyến nghị thử trước trên GPU 16GB.

Diễn giải nhanh:

- `patch_size = 16` nghĩa là ảnh `384x384` được chia thành `24 x 24 = 576` patch.
- `1 frame` sau encoder tương ứng `576 token`.
- `1 sample` hiện tại dùng `8 frame`, nên latent của một sample là `8 x 576 = 4608 token`.
- `tubelet_size = 2` là temporal patching của V-JEPA; trong code RC hiện tại mỗi frame thật được duplicate thành pseudo-clip `2 frame` để encoder trả ra feature cho từng frame riêng.
- ảnh processed hiện được resize offline về `224x224`, sau đó `FrozenVJepa21Encoder` resize lên `384x384` trước khi đưa vào V-JEPA.

Predictor NN-JEPA hiện tại có 2 loại:

```text
predictor_type = simple         baseline cũ, mặc định
predictor_type = official_lite  official-style AC predictor nhẹ, giữ Simple làm baseline
```

`simple` dùng `SimpleACPredictor`:

```text
predictor_dim = 512
predictor_depth = 6
predictor_heads = 8
feedforward_dim = 2048
dropout = 0.0
state_dim = 5
action_dim = 2
cond_tokens_per_frame = 2
max_frames = 8
tokens_per_frame = 576
predictor_params = 20,007,680
```

`official_lite` dùng `VJepaStyleACPredictor`:

```text
token_layout = [action token, state token, patch tokens]
attention_mask = action-block causal attention mask giống source vjepa2
rope = frame/height/width RoPE theo source vjepa2
output = latent patch tokens
state_dim = 5
action_dim = 2
cond_tokens_per_frame = 2
tokens_per_frame = 576
```

Preset predictor hiện có với ViT-B 384 feature:

```text
predictor_type  model_size  predictor_dim  depth  heads  params
simple          tiny        128            2      4      670,464
simple          small       256            4      4      3,706,112
simple          base        512            6      8      20,007,680
official_lite   tiny        128            2      4      595,456
official_lite   small       256            4      4      3,556,096
official_lite   base        512            6      8      19,707,648
```

`simple base` là mặc định và giữ nguyên baseline hiện tại. `tiny` dùng để thử pipeline/loss/W&B/resume nhanh hơn trước khi chạy bản `base` lâu. `official_lite tiny` là cấu hình nên thử đầu tiên nếu muốn tiến gần V-JEPA AC official hơn.

Train NN-JEPA hiện tại:

```text
epochs = 100
batch_size = 10
eval_batch_size = 2
num_workers = 4
optimizer = AdamW
scheduler = torch.optim.lr_scheduler.LambdaLR
lr = 1e-4
weight_decay = 1e-4
grad_clip = 1.0
warmup_epochs = 4
warmup_start_factor = 0.1
min_lr_ratio = 0.1
lr_schedule = linear warmup -> cosine decay
early_stopping_patience = 15
early_stopping_count = bắt đầu sau warmup
auto_steps = 2
loss = teacher_forcing_loss + rollout_loss
loss_type = L1 trên latent token
```

Sequence/data NN-JEPA hiện tại:

```text
raw_frames_per_sample = 8
sequence_stride = 1
max_frame_index_gap = 1
max_time_gap_sec = 0.25
state_columns = [yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]
action_columns = [steering_cmd_t, throttle_cmd_t]
normalize_state_inputs = True
normalize_action_inputs = True
image_normalization = ImageNet mean/std
online_image_augmentation = chỉ có khi train từ ảnh, không có khi train từ feature cache
dataloader_num_workers_default = 4
dataloader_prefetch_factor = 4
dataloader_pin_memory = True
dataloader_persistent_workers = True
```

Đối chiếu paper, config public `vjepa2` và NN-JEPA:

```text
Paper robot manipulation:
- train một action-conditioned predictor trên feature encoder đã pretrained
- loss gồm teacher-forcing và rollout/autoregressive
- predictor robot trong paper lớn hơn NN-JEPA hiện tại nhiều
```

```text
Config public robot AC trong vjepa2/configs/train/vitg16/droid-256px-8f.yaml:
- app = vjepa_droid
- model_name = vit_giant_xformers
- crop_size = 256
- dataset_fpcs = 8
- fps = 4
- batch_size = 8
- epochs = 315
- warmup = 15
- lr = 0.000425
- start_lr = 0.000075
- weight_decay = 0.04
- final_weight_decay = 0.04
- patch_size = 16
- tubelet_size = 2
- auto_steps = 2
- pred_depth = 24
- pred_embed_dim = 1024
- pred_num_heads = 16
- pred_is_frame_causal = true
- use_extrinsics = false
- normalize_reps = true
- loss_exp = 1.0
- dtype = bfloat16
- pretrain_checkpoint = /your_vjepa2_checkpoints/vitg.pt
- context_encoder_key = target_encoder
- target_encoder_key = target_encoder
- load_predictor = false
```

```text
V-JEPA 2.1 pretrain config local, ví dụ vjepa2/configs/train_2_1/vitb16/pretrain-256px-16f.yaml:
- app = vjepa_2_1
- model_name = vit_base
- crop_size = 256
- patch_size = 16
- tubelet_size = 2
- dataset_fpcs = 16
- fps = 4
- video batch_size = 48
- image batch_size = 144
- pred_depth = 12
- pred_embed_dim = 384
- pred_num_heads = 12
- epochs = 1000
- warmup = 40
- ipe = 300
- ipe_scale = 1.25
- lr = 0.0006
- start_lr = 0.0001
- weight_decay = 0.04
- dtype = bfloat16
```

```text
V-JEPA 2.1 cooldown config local, ví dụ vjepa2/configs/train_2_1/vitb16/cooldown-256px-64f.yaml:
- model_name = vit_base
- crop_size = 256
- patch_size = 16
- tubelet_size = 2
- dataset_fpcs = 64
- video batch_size = 24
- image batch_size = 144
- epochs = 40
- warmup = 0
- lr = 0.0006
- final_lr = 1e-6
- anneal_ckpt trỏ về checkpoint pretrain
```

Khác biệt quan trọng:

- NN-JEPA không train lại encoder; encoder được freeze.
- NN-JEPA hiện dùng `vit_base_384` nhẹ hơn rất nhiều so với public robot config `vit_giant_xformers`.
- NN-JEPA predictor là bản nhỏ `20.01M params`; public robot config dùng predictor lớn `24 layer`, `1024 dim`.
- NN-JEPA không dùng extrinsics/camera pose vì xe RC hiện chưa có camera extrinsics ổn định như DROID.
- NN-JEPA dùng state/action xe RC, không dùng state/action 7 chiều của robot DROID.

Với manifest hiện tại và `batch_size=10`:

```text
train_samples = 84,766
val_samples = 29,780
test_samples = 29,780  # alias của val, không phải test độc lập
train_windows = 68,139
val_windows = 24,397
test_windows = 24,397  # alias của val, không phải test độc lập
steps_per_epoch = 6,814
warmup_steps = 27,256
```

Diễn giải LR hiện tại:

- LR base: `1e-4`
- scheduler dùng PyTorch `LambdaLR`
- warmup bắt đầu từ `1e-5` (`0.1 * 1e-4`)
- tăng tuyến tính lên `1e-4` trong `4 epoch`
- sau đó giảm bằng `cosine decay`
- LR cuối sẽ về khoảng `1e-5`
- nếu resume từ checkpoint cũ chưa có `lr_scheduler_state_dict`, code vẫn dựng lại scheduler từ `global_step`, nên không cần train lại từ đầu

Resume train:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac \
  --resume-from checkpoints/rc_jepa_ac/last.pt \
  --epochs 100 \
  --batch-size 10 \
  --lr 1e-4
```

Các file sẽ xuất hiện trong `output-dir`:

```text
run_config.json
history.json
last.pt
best.pt
final_metrics.json
epochs/epoch_001.pt
epochs/epoch_002.pt
...
```

### 6. Extract feature trước rồi train predictor nhanh hơn

Nếu muốn train nhanh hơn, chạy encoder V-JEPA 2.1 một lần để lưu latent frame ra disk, sau đó train predictor từ latent đã cache.

Lệnh extract feature:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --batch-size 32 \
  --dtype fp16
```

Preset V-JEPA 2.1 cho feature extractor:

```text
preset    encoder              checkpoint default                                      key
vitb_384  vit_base_384         vjepa2_1_vitb_dist_vitG_384.pt                          ema_encoder
vitl_384  vit_large_384        vjepa2_1_vitl_dist_vitG_384.pt                          ema_encoder
vitg_384  vit_giant_384        vjepa2_1_vitg_384.pt                                    target_encoder
vitG_384  vit_gigantic_384     vjepa2_1_vitG_384.pt                                    target_encoder
```

Mặc định `--output-dir` sẽ tự đổi theo preset và dtype, ví dụ:

```text
vitb_384 fp16 -> data/processed/features/vjepa2_1_vitb_384_ema_fp16
vitl_384 fp16 -> data/processed/features/vjepa2_1_vitl_384_ema_fp16
vitg_384 fp16 -> data/processed/features/vjepa2_1_vitg_384_target_fp16
vitG_384 fp16 -> data/processed/features/vjepa2_1_vitG_384_target_fp16
```

Lưu ý: nếu đổi preset encoder thì phải train predictor lại từ đầu với đúng `--features-dir` mới. Predictor train từ ViT-B feature không dùng được trực tiếp cho ViT-L/ViT-g/ViT-G feature vì `embed_dim` khác.

Lệnh train predictor từ feature cache, cấu hình khuyến nghị hiện tại để giữ chất lượng:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607 \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4 \
  --warmup-epochs 4 \
  --warmup-start-factor 0.1 \
  --min-lr-ratio 0.1 \
  --early-stopping-patience 15 \
  --wandb-log-every 20 \
  --wandb-watch-log all \
  --wandb-watch-freq 100 \
  --wandb-grad-stats-every 10 \
  --wandb-param-stats-every 100
```

Mặc định `tools.train_rc_jepa_ac_features` bỏ phase `test` cuối train. Script chỉ chạy `val` sau mỗi epoch, chọn `best.pt` theo `val/loss`, rồi ghi `final_metrics.json`. Vì hiện tại không có test split độc lập, `test.jsonl` chỉ là alias của `val.jsonl`; nếu thêm `--run-test` thì kết quả `test/*` cũng chính là đo lại trên val.

```bash
--run-test
```

Lệnh train bản `tiny` để thử nghiệm nhanh trước:

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
  --warmup-start-factor 0.1 \
  --min-lr-ratio 0.1 \
  --early-stopping-patience 5 \
  --wandb-project nn-jepa-rc \
  --wandb-tags tiny
```

Lưu ý cho bản `tiny`:

- vẫn dùng cùng feature cache `fp16`, cùng sample `8 frame`, cùng loss `teacher_forcing_loss + rollout_loss`
- chỉ giảm predictor từ `20,007,680` xuống `670,464` tham số
- nếu không truyền `--output-dir`, script tự đổi default sang `checkpoints/rc_jepa_ac_vitb_features_20260607_tiny`
- dùng để debug pipeline/hyperparameter/logging; khi đã ổn thì chạy lại bản `base`

Lệnh train bản `official_lite tiny` để thử predictor gần V-JEPA AC official hơn:

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
  --warmup-start-factor 0.1 \
  --min-lr-ratio 0.1 \
  --early-stopping-patience 5 \
  --wandb-project nn-jepa-rc \
  --wandb-tags official_lite tiny
```

Hoặc dùng Hydra:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny
```

Lưu ý cho `official_lite`:

- giữ full token `576 token/frame`, không pooling
- layout token là `[action, state, patch tokens]`
- mask causal dạng action-block khớp source `vjepa2`
- nên bắt đầu với `batch_size=4`, `eval_batch_size=1` để tránh OOM
- checkpoint/eval/infer tự đọc `predictor_type`, nên checkpoint `official_lite` sẽ rebuild đúng model

Cấu hình trên giữ:

- `fp16` latent cache để giảm dung lượng disk; Dataset vẫn convert latent sang `float32` trước khi đưa vào predictor.
- full patch token `576 token/frame`, không pooling, không giảm token.
- `train batch_size = 10`, `val batch_size = 2`, predictor mặc định `20.01M params`.
- W&B log đầy đủ loss, gradient, parameter histogram và gradient scalar stats.
- mặc định script hiện dùng `num_workers = 4`, `prefetch_factor = 4`

Nếu GPU vẫn hay chờ data, tăng thêm worker:

```bash
--num-workers 8
```

Nếu W&B logging làm train chậm hoặc dễ đẩy VRAM lên cao hơn ở pha `val`, chỉ giảm logging, không đổi model/data:

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
  --wandb-log-every 20 \
  --wandb-watch-log gradients \
  --wandb-watch-freq 200 \
  --wandb-grad-stats-every 20 \
  --wandb-param-stats-every 200
```

Điểm chính của lệnh trên:

- dùng feature cache `fp16`, giữ full token `576 token/frame`, predictor mặc định `20.01M params`
- không giảm `train batch_size`, chỉ giữ `val` xuống `2` cho an toàn hơn
- chỉ giảm tải W&B và thêm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` để giảm rủi ro phân mảnh VRAM

Resume train từ feature cache:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607 \
  --resume-from checkpoints/rc_jepa_ac_vitb_features_20260607/last.pt \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4
```

Checkpoint của `train_rc_jepa_ac_features` hiện có 2 mức resume:

- `last.pt`: lưu sau khi xong cả `train` và `val` của một epoch
- `last_train.pt`: lưu ngay sau khi xong `train` của epoch, trước khi bước vào `val`

Nếu run bị dừng giữa `val`, đặc biệt là OOM ở `val`, resume bằng:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607 \
  --resume-from checkpoints/rc_jepa_ac_vitb_features_20260607/last_train.pt \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4
```

Script sẽ nhận ra checkpoint đang ở pha `train_complete_waiting_val` và đi tiếp vào `val` của đúng epoch đó, không train lại epoch vừa xong.

W&B resume cùng một run:

- khi train lần đầu, script tự lưu W&B run id vào `checkpoints/rc_jepa_ac_vitb_features_20260607/wandb_run_id.txt`
- khi chạy lại với cùng `--output-dir` và có `--resume-from`, script tự đọc file này và gọi W&B với `id=<run_id>` + `resume="allow"`
- nghĩa là lệnh resume ở trên sẽ nối log vào cùng run W&B, không tự tách run mới nữa
- khi resume cùng run, giữ cùng `--wandb-project` và `--wandb-entity` với run gốc
- nếu muốn resume checkpoint nhưng cố tình tạo W&B run mới, thêm `--no-wandb-continue-run`

Với Hydra:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.resume_from=checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/last.pt \
  wandb.continue_run=false
```

Nếu run W&B cũ được tạo trước khi có file `wandb_run_id.txt`, lấy run id ở cuối URL W&B rồi truyền thủ công:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607 \
  --resume-from checkpoints/rc_jepa_ac_vitb_features_20260607/last.pt \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4 \
  --wandb-run-id hei7na3j \
  --wandb-resume allow
```

Chạy standalone eval/test từ checkpoint:

```bash
PYTHONPATH=src python3 -m tools.eval_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --split test \
  --eval-batch-size 2 \
  --num-workers 8
```

Kết quả sẽ được ghi vào:

```text
checkpoints/rc_jepa_ac_vitb_features_20260607/eval_test.json
```

Nếu muốn eval cả `train`, `val`, `test`:

```bash
--split all
```

Lưu ý: `--split test` hiện không phải test độc lập. Nó đọc `test.jsonl`, nhưng file này là alias của `val.jsonl`, nên chỉ dùng để tương thích quy trình cũ hoặc để tạo file kết quả tên `eval_test.json`.

Chạy inference từ checkpoint:

```bash
PYTHONPATH=src python3 -m tools.infer_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features_20260607/best.pt \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --split test \
  --max-samples 32 \
  --eval-batch-size 2 \
  --num-workers 8
```

Inference mặc định ghi metric gọn theo sample vào:

```text
checkpoints/rc_jepa_ac_vitb_features_20260607/inference/inference_test.jsonl
```

Nếu muốn lưu cả tensor latent dự đoán và target cho từng sample, thêm:

```bash
--save-tensors
```

Lưu ý: inference hiện tại là inference của **world model latent**, tức là dự đoán latent frame tương lai từ latent hiện tại + state + action. Nó không sinh trực tiếp `steering_cmd_t`/`throttle_cmd_t`.

Chạy planner offline kiểu JEPA để sinh action bằng CEM:

```bash
PYTHONPATH=src python3 -m tools.plan_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/best.pt \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --manifest-dir data/processed/manifests \
  --split test \
  --max-samples 32 \
  --horizon 2 \
  --goal-offset 2 \
  --cem-samples 128 \
  --cem-elites 16 \
  --cem-iters 4 \
  --num-workers 8
```

Nếu checkpoint mới chưa train xong và chỉ muốn smoke/demo bằng checkpoint cũ đang có trên máy, đổi checkpoint thành:

```bash
--checkpoint checkpoints/rc_jepa_ac_vitb_features/best.pt
```

Lưu ý: checkpoint cũ có thể chạy được nếu feature shape khớp, nhưng không nên dùng kết quả đó để kết luận chất lượng trên data mới.

Planner offline sẽ:

- lấy latent frame đầu `z_t` từ feature cache
- lấy goal latent `z_{t+k}` trong cùng sample
- sample nhiều chuỗi action raw `[steering_cmd_t, throttle_cmd_t]`
- normalize action đúng như lúc train trước khi đưa vào predictor
- rollout predictor để dự đoán latent tương lai
- chọn chuỗi action có latent cuối gần goal latent nhất
- ghi kết quả vào `checkpoints/.../planning/planning_test.jsonl` và `planning_test.csv`

Ý nghĩa output planner:

```text
planned_first_*        action đầu tiên CEM muốn chạy
groundtruth_first_*    action thật trong log tại cùng frame
planned_final_l1       latent L1 tới goal khi dùng action planner
groundtruth_final_l1   latent L1 tới goal khi dùng action thật
zero_action_final_l1   latent L1 tới goal nếu giữ action bằng 0
```

Không nên hiểu `planned_first_*` bắt buộc phải giống ground-truth. Cùng một goal latent có thể đạt bằng nhiều action sequence khác nhau. Chỉ số cần xem trước là `planned_final_l1` có hợp lý và có tốt hơn `zero_action_final_l1` không.

Vẽ đồ thị từ kết quả planner:

```bash
PYTHONPATH=src python3 -m tools.plot_rc_jepa_planning \
  --planning-jsonl checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/planning/planning_test.jsonl
```

Tool này không cần `matplotlib`; nó ghi SVG vào:

```text
checkpoints/rc_jepa_ac_vitb_features_newdata_tiny/planning/plots/
```

Các đồ thị chính:

- `latent_l1_comparison.svg`: so sánh `planned_final_l1`, `groundtruth_final_l1`, `zero_action_final_l1`.
- `first_action_planned_vs_groundtruth.svg`: so sánh action đầu tiên planner sinh với action thật.
- `first_action_abs_error.svg`: sai số tuyệt đối của action đầu tiên.
- `action_sequence_record_*.svg`: chuỗi action planner vs ground-truth theo từng sample.

Lưu ý an toàn về inference:

- `tools.infer_rc_jepa_ac_features` chỉ kiểm tra dự đoán latent theo action có sẵn trong data.
- `tools.plan_rc_jepa_ac_features` mới là bước sinh action offline bằng world model + planner.
- Đây vẫn chưa phải closed-loop thật; script chưa gửi lệnh xuống xe.
- Trước khi closed-loop cần thêm live dry-run, watchdog, neutral fallback, clamp throttle/steering và kênh gửi lệnh phần cứng ổn định.
- Không trộn feature/checkpoint JEPA `ViT-L 256, 256 token/frame, D=1024` với NN-JEPA `ViT-B 384, 576 token/frame, D=768`.

Lưu ý khi dùng feature cache:

- bước extract vẫn cần checkpoint encoder V-JEPA 2.1
- bước train từ feature không load encoder nữa, chỉ train predictor
- `tools.train_rc_jepa_ac_features` đọc `latents` từ `.npy`, không gọi `FrozenVJepa21Encoder`
- `tools.extract_vjepa_features` skip session đã có `.npy + .json` đúng shape/dtype, không extract lại
- nếu toàn bộ cache đã đầy đủ và metadata khớp, extractor kết thúc sớm mà không load encoder/checkpoint
- trước `val` và `test`, script gọi `torch.cuda.empty_cache()` để giảm bớt rủi ro OOM do bộ nhớ đệm cũ
- `checkpoint_key = ema_encoder` là key lấy weight encoder EMA trong checkpoint, không phải predictor
- không lấy key `predictor` vì predictor trong checkpoint V-JEPA là predictor pretrain, không phải action-conditioned predictor cho xe RC
- train từ cache nhanh hơn nhưng không còn augmentation ảnh online, vì ảnh đã được encode cố định
- train từ cache vẫn nặng vì một sample có `8 x 576 = 4608 token`, batch `10` là khoảng `46k token` trước khi qua transformer predictor

Vì sao hiện chuyển mặc định sang `fp16`?

- `fp32` giữ latent cache chính xác hơn `fp16`, nhưng với data hiện tại tốn khoảng `265 GiB`.
- `fp16` giảm khoảng một nửa dung lượng, còn khoảng `133 GiB`.
- Khi train, Dataset đọc cache và convert latent lại về `float32` trước khi đưa vào predictor.
- Đánh đổi chính là latent đã lưu có precision thấp hơn fp32, nhưng đây là lựa chọn thực tế hơn khi disk đang quá nặng.
- Train online trước khi trích xuất feature cũng chạy `float32` trong NN-JEPA vì code không bật AMP/autocast.

Dung lượng feature cache với data hiện tại:

```text
manifest frame_count = 161,037
tokens_per_frame = 576
embed_dim = 768
per_frame_fp16 = 0.84375 MiB
per_frame_fp32 = 1.6875 MiB
total_fp16 ~= 132.69 GiB
total_fp32 ~= 265.38 GiB
```

Lệnh extract `fp16` nếu muốn tiết kiệm dung lượng:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --batch-size 32 \
  --dtype fp16
```

Nếu đổi `--dtype`, `--encoder-preset`, `--encoder`, `--image-size`, `--patch-size` hoặc `--tubelet-size`, nên dùng output dir riêng. Khi không truyền `--output-dir`, script sẽ tự chọn dir riêng theo preset/dtype. Script extract cũng kiểm tra shape/dtype của cache cũ; nếu không khớp nó sẽ báo lỗi thay vì âm thầm dùng nhầm feature.

### 7. Experiment dùng thêm data servo cũ

Experiment servo cũ được tách riêng khỏi baseline. Không copy data cũ vào `data/raw` và không ghi đè `data/processed`.

Data servo cũ hiện nằm ở:

```text
JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV/
```

Build mixed dataset, gồm `data/raw` hiện tại + servo cũ:

```bash
PYTHONPATH=src python3 -m tools.build_servo_experiment_dataset \
  --mode mixed \
  --experiment-root data/experiments/servo_old_mix_v1 \
  --split-file data/split_vjepa_ac_car.json \
  --no-test-split
```

`--no-test-split` là default hiện tại. Khi không chia test độc lập, tool vẫn ghi `test.jsonl` bằng chính `val.jsonl` để các lệnh `--split test` cũ không bị hỏng.

Hai hướng experiment hiện tại:

- Không trộn old-servo: dùng baseline `data/processed/manifests` và feature cache `data/processed/features/vjepa2_1_vitb_384_ema_fp16`.
- Có trộn old-servo: dùng `servo_old_mix_v1`, manifest riêng ở `data/experiments/servo_old_mix_v1/processed/manifests`, split theo `data/split_vjepa_ac_car.json`.

Trạng thái hiện tại trên disk:

- baseline `data/raw` có `181` session current.
- `servo_old_mix_v1` đã rebuild theo split JSON: `181 current_servo + 30 old_servo = 211` session.
- `servo_old_only_v1` đã bị xóa và config old-only cũng đã bỏ.

Sau khi GPU hoạt động, extract feature cho mixed experiment:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --splits train val \
  --output-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --seed-from-features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 32 \
  --dtype fp16 \
  --splits train val
```

`--seed-from-features-dir` sẽ reuse feature baseline fp16 nếu metadata encoder khớp, rồi chỉ encode thêm phần old-servo chưa có cache.

Train mixed tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2
```

Config mixed đang dùng `simple tiny`, `raw_frames_per_sample=8`, `frame_stride=2`, `batch_size=32`, `eval_batch_size=2`. Nên train fresh, không resume checkpoint cũ, vì data distribution và split đã đổi.

Báo cáo chi tiết:

```text
doc/bao_cao_trien_khai_experiment_servo_cu.md
```

## State và action hiện tại

Schema đầy đủ đang bám theo mục tiêu ban đầu:

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

Tuy nhiên trong bản `rc_jepa_ac` hiện tại, state mặc định đang dùng là:

```text
[
  yaw_rate_t,
  accel_x_t,
  accel_y_t,
  steering_last_t,
  throttle_last_t,
]
```

`v_t` đang tạm bỏ khỏi default vì nguồn hiện tại là `gps.speed`, chạy trong nhà thường nhiễu.

## Các bảo vệ train hiện tại

Pipeline hiện đã có các lớp kiểm tra cơ bản trước khi train:

- `REMOVE_SIMPLE_OUTLIERS = True`: preprocess loại spike sensor bằng median/MAD robust, tránh các điểm IMU/GPS lệch quá mạnh.
- `AC_MAX_FRAME_INDEX_GAP = 1`: dataset sequence không tạo sample nếu frame bị đứt quãng.
- `AC_MAX_TIME_GAP_SEC = 0.25`: dataset sequence không nối các frame cách nhau quá lâu.
- `NORMALIZE_STATE_INPUTS = True`: state input được chuẩn hóa bằng thống kê từ `train.jsonl`.
- `NORMALIZE_AC_ACTION_INPUTS = True`: action input của world model được chuẩn hóa bằng thống kê train.
- `rc_jepa_ac` rollout không dùng state tương lai thật; các bước rollout chỉ dùng state ban đầu và action đã biết.

Sau lần preprocess baseline gần nhất:

```text
raw_sessions: 181
train: 103208 samples, 82623 windows
val:   57829 samples, 48103 windows
test:  57829 samples, 48103 windows  # alias của val
```

Report chi tiết nằm ở:

```text
data/processed/reports/preprocess_report.json
```

## Hiện trạng đã kiểm tra

Lần kiểm tra gần nhất:

```text
zip top-level trong `JEPA/data/drive_zips`: 181
session zip tổng cộng trong `JEPA/data/drive_zips/**`: 183
data servo cũ KDS 680HV: 30 session, 56,023 file
old-servo extra từ `trong nhà`: 2 session đã giải nén thêm để khớp split JSON
raw train sessions: 181
raw session cũ 20260605 trong data/raw: 0
actions_synced.csv / imu_synced.csv thiếu: 0
manifest missing frame_path: 0
train/val split độc lập theo session; test split không độc lập vì test.jsonl là alias của val.jsonl
feature cache baseline fp32: đã xóa
feature cache baseline fp16: chưa extract lại
servo_old_mix_v1: 211 session selected = 181 current + 30 old
servo_old_mix_v1 samples train/val/test: 182,562 / 30,282 / 30,282 alias val
servo_old_only_v1: đã xóa
servo_old frame_stride=2 windows:
  mixed train/val/test = 128,617 / 20,501 / 20,501 alias val
web smoke test: /, /app.js, /styles.css, /api/sessions, /api/jobs đều HTTP 200
unit tests: 47/47 pass
```

Diễn giải `181` và `183`:

- `181` là số zip top-level hiện đang được staging trong `JEPA/data/drive_zips`, tương ứng với `181` session current đang có trong `data/raw`.
- `183` là tổng số file `session_*.zip` nếu tính cả zip nằm trong thư mục con; chênh `2` là các zip cũ không được đưa vào baseline train hiện tại.

Data servo cũ:

- Đường dẫn: `JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV/`.
- Đây là data cũ `20260605`, chỉ giữ để tham khảo/backup, không đưa vào `data/raw` và không dùng train hiện tại.
- Đã khôi phục từ 3 zip thủ công:
  `data servo cũ KDS 680HV-20260608T112340Z-3-001.zip`,
  `data servo cũ KDS 680HV-20260608T112340Z-3-002.zip`,
  `data servo cũ KDS 680HV-20260608T112340Z-3-003.zip`.
- Sau giải nén có `28` session, `53,403` file.
- Session `session_20260605_155710` từng thiếu synced; đã chạy lại `jepa_wm.data.sync`, giữ `47` frame, bỏ `1`, `offset=100ms +imu`.
- Hiện `actions_synced.csv` và `imu_synced.csv` đủ `28/28`.

Lệnh kiểm tra zip với Drive:

```bash
conda run -n nn-jepa rclone check gdrive:JEPA JEPA/data/drive_zips \
  --include '*.zip' \
  --one-way \
  --fast-list
```

Lệnh kiểm tra data/feature nhanh:

```bash
PYTHONPATH=src python3 - <<'PY'
import json
from pathlib import Path

raw = Path("data/raw")
features = Path("data/processed/features/vjepa2_1_vitb_384_ema_fp16")
metadata = json.loads((features / "metadata.json").read_text())
raw_sessions = [p for p in raw.glob("session_*") if p.is_dir()]
print("raw_sessions", len(raw_sessions))
print("old_20260605_raw", len([p for p in raw_sessions if p.name.startswith("session_20260605_")]))
print("missing_synced", len([
    p for p in raw_sessions
    if not (p / "actions_synced.csv").exists() or not (p / "imu_synced.csv").exists()
]))
print("feature_npy", len(list((features / "sessions").glob("*.npy"))))
print("feature_json", len(list((features / "sessions").glob("*.json"))))
print("feature_metadata", metadata["session_count"], metadata["frame_count"], metadata["dtype"])
PY
```

Lệnh kiểm tra data servo cũ:

```bash
python3 - <<'PY'
from pathlib import Path

root = Path("JEPA/data/drive_extra_nonzip/data servo cũ KDS 680HV")
sessions = [p for p in root.glob("session_*") if p.is_dir()]
missing = []
for session in sessions:
    for name in ("actions_synced.csv", "imu_synced.csv"):
        if not (session / name).exists():
            missing.append(f"{session.name}/{name}")
print("servo_old_sessions", len(sessions))
print("servo_old_files", len([p for p in root.rglob("*") if p.is_file()]))
print("missing_synced", missing)
PY
```

## Các file chính

```text
src/data/settings.py
src/data/preprocess.py
src/data/dataset.py
src/data/sequence_dataset.py
src/data/feature_sequence_dataset.py
src/models/rc_car_model.py
src/models/rc_jepa_ac.py
src/tools/preprocess_data.py
src/tools/sync_drive_data.py
src/tools/export_session_gif.py
src/tools/session_web_viewer.py
src/tools/train_rc_car.py
src/tools/train_rc_jepa_ac.py
src/tools/extract_vjepa_features.py
src/tools/train_rc_jepa_ac_features.py
src/tools/eval_rc_jepa_ac_features.py
src/tools/infer_rc_jepa_ac_features.py
src/tools/plan_rc_jepa_ac_features.py
src/tools/rc_jepa_ac_cem_planner.py
src/tools/rc_jepa_ac_feature_runtime.py
```

## Chỗ chỉnh nhanh nhất

Muốn đổi biến toàn cục cho pipeline thì sửa ở:

```text
src/data/settings.py
```

Các nhóm biến chính:

- đường dẫn data
- kích thước ảnh
- split train / val; `test.jsonl` là alias của val để tương thích tool cũ
- stride lấy frame
- tên cột state / action
- ngưỡng outlier
- ngưỡng frame/time gap cho sequence
- bật/tắt normalize numeric input
- augmentation
- tolerance khi match sensor

## Kiểm tra nhanh

Compile:

```bash
/home/heheboiz/miniconda3/envs/nn-jepa/bin/python -m compileall src tests
```

Chạy test:

```bash
PYTHONPATH=src /home/heheboiz/miniconda3/envs/nn-jepa/bin/python -m unittest discover -s tests -v
```

Smoke test web API, nếu muốn kiểm tra server:

```bash
PYTHONPATH=src python3 -m tools.session_web_viewer
```

Mở hoặc kiểm tra:

```text
http://127.0.0.1:8765/api/sessions?source=raw
http://127.0.0.1:8765/api/jobs
```

## Progress Bar Khi Train

`tools.train_rc_car`, `tools.train_rc_jepa_ac` và `tools.train_rc_jepa_ac_features` đều chạy `val` ngay sau mỗi epoch và hiển thị progress bar bằng `tqdm`.

Riêng `tools.train_rc_jepa_ac_features` mặc định bỏ phase `test` cuối train; chỉ chạy `train` và `val`. Muốn chạy test cuối train thì thêm `--run-test`, hoặc chạy `tools.eval_rc_jepa_ac_features` riêng sau khi có `best.pt`.

Nếu muốn tắt progress bar:

```bash
--no-progress
```

## Weights & Biases

Các script train log lên W&B mặc định:

```text
project: nn-jepa-rc
metrics: train/*, val/*, test/*, best/val_loss, lr
```

Tất cả metric/loss mà loop đang trả về đều được log lên W&B.

Với `tools.train_rc_jepa_ac_features`, mặc định chỉ có `train/*`, `train_batch/*`, `val/*`, `best/*`, `lr`. `test/*` chỉ có nếu bật `--run-test` hoặc chạy eval/test riêng.

Ngoài metric theo epoch, loop train còn log thêm `train_batch/*` theo batch để nhìn thấy loss curve ngay trong lúc epoch đang chạy.

Trục step trên W&B:

- `train_batch/*` log theo `global_step`.
- `train/*`, `val/*`, `best/*` cũng log theo `global_step`.
- `test/*` chỉ log khi có chạy test.
- Không dùng `epoch` làm W&B step, để tránh trường hợp step đi lùi và W&B bỏ qua metric.
- Metric vẫn có field `epoch`, nên vẫn lọc/xem theo epoch được.

Gradient/parameter logging mặc định:

```text
wandb_watch_log = gradients
wandb_watch_freq = 200
wandb_grad_stats_every = 20
wandb_param_stats_every = 200
```

Các metric gradient/parameter thêm trên W&B:

```text
train_batch/grad_pre_clip/global_l2
train_batch/grad_pre_clip/mean_abs
train_batch/grad_pre_clip/max_abs
train_batch/grad_pre_clip/nonfinite_count
train_batch/grad_pre_clip/zero_value_count
train_batch/grad_pre_clip/tensor_count
train_batch/grad_pre_clip/value_count
train_batch/grad_pre_clip/missing_tensor_count
train_batch/grad_pre_clip_norm/latent_proj
train_batch/grad_pre_clip_norm/state_proj
train_batch/grad_pre_clip_norm/action_proj
train_batch/grad_pre_clip_norm/blocks
train_batch/grad_pre_clip_norm/output_proj
train_batch/grad_post_clip/global_l2
train_batch/grad_post_clip/mean_abs
train_batch/grad_post_clip/max_abs
train_batch/grad_clip/pre_clip_global_l2
train_batch/grad_clip/max_norm
train_batch/param/global_l2
train_batch/param/mean_abs
train_batch/param/max_abs
train_batch/param_norm/blocks
```

Nếu muốn log histogram gradient và parameter nhiều hơn:

```bash
--wandb-watch-log all \
--wandb-watch-freq 100 \
--wandb-grad-stats-every 10 \
--wandb-param-stats-every 100
```

Nếu W&B làm train chậm hoặc log quá nặng:

```bash
--wandb-watch-log none \
--wandb-grad-stats-every 0 \
--wandb-param-stats-every 0
```

Ví dụ với `rc_jepa_ac`:

```text
train/loss
train/teacher_forcing_loss
train/rollout_loss
val/loss
val/teacher_forcing_loss
val/rollout_loss
test/loss
test/teacher_forcing_loss
test/rollout_loss
```

Ví dụ với `rc_car`:

```text
train/loss
train/steering_mae
train/throttle_mae
val/loss
val/steering_mae
val/throttle_mae
test/loss
test/steering_mae
test/throttle_mae
```

Trước khi train, login một lần:

```bash
wandb login
```

Muốn ép run vào đúng account hoặc team, truyền rõ `--wandb-entity`. Nếu bỏ trống, W&B sẽ dùng entity mặc định của tài khoản đang đăng nhập.

Ví dụ đặt tên run:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint /duong/dan/toi/checkpoint.pt \
  --wandb-project nn-jepa-rc \
  --wandb-entity ten-user-hoac-team \
  --wandb-run-name rc-jepa-ac-smoke-001
```

Resume W&B vào cùng run:

- `--wandb-run-id`: id ở cuối URL run, ví dụ URL `.../runs/hei7na3j` thì run id là `hei7na3j`
- `--wandb-resume allow`: nối vào run cũ nếu đã tồn tại, còn nếu chưa tồn tại thì W&B vẫn cho tạo
- mặc định script train tự lưu run id vào `wandb_run_id.txt` trong `--output-dir`, nên resume checkpoint bằng cùng `--output-dir` thường không cần truyền `--wandb-run-id` thủ công
- khi nối run cũ, dùng đúng `--wandb-project` và `--wandb-entity` của run đó

Nếu muốn chạy không gửi lên W&B:

```bash
--no-wandb
```

Nếu muốn log offline rồi sync sau:

```bash
--wandb-mode offline
```
