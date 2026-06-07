# NN-JEPA

Repo này là phần **train** cho xe RC. Repo `JEPA/` chỉ phục vụ phần cứng, recorder, sync.

Luồng hiện tại:

```text
data/raw/session_xxx/... -> preprocess -> data/processed/... -> train / viewer
```

## Cấu trúc dữ liệu

Raw data nằm trong:

```text
data/raw/
  session_20260605_150919/
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

### 1. Preprocess lại dữ liệu

```bash
PYTHONPATH=src python3 -m tools.preprocess_data
```

Pipeline sẽ:

- đọc toàn bộ session trong `data/raw`
- resize ảnh
- ghi ảnh processed vào `data/processed/images`
- tạo manifest train/val/test
- ghi report vào `data/processed/reports/preprocess_report.json`

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

### 3. Export nhanh 1 session ra GIF

```bash
PYTHONPATH=src python3 -m tools.export_session_gif \
  --session-id session_20260605_150919
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
- lịch LR hiện tại: `linear warmup 4 epoch` rồi `cosine decay`
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
name           params       embed_dim  depth  heads  image  patch  tubelet  tokens/frame
vit_small_384  22,182,912   384        12     6      384    16     2        576
vit_base_384   86,833,152   768        12     12     384    16     2        576
vit_large_384  304,680,960  1024       24     16     384    16     2        576
```

Diễn giải nhanh:

- `patch_size = 16` nghĩa là ảnh `384x384` được chia thành `24 x 24 = 576` patch.
- `1 frame` sau encoder tương ứng `576 token`.
- `1 sample` hiện tại dùng `8 frame`, nên latent của một sample là `8 x 576 = 4608 token`.
- `tubelet_size = 2` là temporal patching của V-JEPA; trong code RC hiện tại mỗi frame thật được duplicate thành pseudo-clip `2 frame` để encoder trả ra feature cho từng frame riêng.
- ảnh processed hiện được resize offline về `224x224`, sau đó `FrozenVJepa21Encoder` resize lên `384x384` trước khi đưa vào V-JEPA.

Predictor NN-JEPA hiện tại:

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

Train NN-JEPA hiện tại:

```text
epochs = 100
batch_size = 10
eval_batch_size = 2
num_workers = 4
optimizer = AdamW
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

Với data hiện tại và `batch_size=10`:

```text
steps_per_epoch = 2345
warmup_steps = 9380
```

Diễn giải LR hiện tại:

- LR base: `1e-4`
- warmup bắt đầu từ `1e-5` (`0.1 * 1e-4`)
- tăng tuyến tính lên `1e-4` trong `4 epoch`
- sau đó giảm bằng `cosine decay`
- LR cuối sẽ về khoảng `1e-5`

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
test_metrics.json
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
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --batch-size 32 \
  --dtype fp32
```

Lệnh train predictor từ feature cache, cấu hình khuyến nghị hiện tại để giữ chất lượng:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features \
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

Cấu hình trên giữ:

- `fp32` latent cache để tránh giảm precision do lưu feature.
- full patch token `576 token/frame`, không pooling, không giảm token.
- `train batch_size = 10`, `val/test batch_size = 2`, predictor mặc định `20.01M params`.
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
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features \
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

- vẫn giữ `fp32`, full token `576 token/frame`, predictor mặc định `20.01M params`
- không giảm `train batch_size`, chỉ giảm `val/test` xuống `2` cho an toàn hơn
- chỉ giảm tải W&B và thêm `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` để giảm rủi ro phân mảnh VRAM

Resume train từ feature cache:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features \
  --resume-from checkpoints/rc_jepa_ac_vitb_features/last.pt \
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
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features \
  --resume-from checkpoints/rc_jepa_ac_vitb_features/last_train.pt \
  --epochs 100 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --lr 1e-4
```

Script sẽ nhận ra checkpoint đang ở pha `train_complete_waiting_val` và đi tiếp vào `val` của đúng epoch đó, không train lại epoch vừa xong.

Chạy standalone eval/test từ checkpoint:

```bash
PYTHONPATH=src python3 -m tools.eval_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features/best.pt \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --split test \
  --eval-batch-size 2 \
  --num-workers 8
```

Kết quả sẽ được ghi vào:

```text
checkpoints/rc_jepa_ac_vitb_features/eval_test.json
```

Nếu muốn eval cả `train`, `val`, `test`:

```bash
--split all
```

Chạy inference từ checkpoint:

```bash
PYTHONPATH=src python3 -m tools.infer_rc_jepa_ac_features \
  --checkpoint checkpoints/rc_jepa_ac_vitb_features/best.pt \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --split test \
  --max-samples 32 \
  --eval-batch-size 2 \
  --num-workers 8
```

Inference mặc định ghi metric gọn theo sample vào:

```text
checkpoints/rc_jepa_ac_vitb_features/inference/inference_test.jsonl
```

Nếu muốn lưu cả tensor latent dự đoán và target cho từng sample, thêm:

```bash
--save-tensors
```

Lưu ý: inference hiện tại là inference của **world model latent**, tức là dự đoán latent frame tương lai từ latent hiện tại + state + action. Nó không sinh trực tiếp `steering_cmd_t`/`throttle_cmd_t`.

Lưu ý khi dùng feature cache:

- bước extract vẫn cần checkpoint encoder V-JEPA 2.1
- bước train từ feature không load encoder nữa, chỉ train predictor
- `tools.train_rc_jepa_ac_features` đọc `latents` từ `.npy`, không gọi `FrozenVJepa21Encoder`
- trước `val` và `test`, script gọi `torch.cuda.empty_cache()` để giảm bớt rủi ro OOM do bộ nhớ đệm cũ
- `checkpoint_key = ema_encoder` là key lấy weight encoder EMA trong checkpoint, không phải predictor
- không lấy key `predictor` vì predictor trong checkpoint V-JEPA là predictor pretrain, không phải action-conditioned predictor cho xe RC
- train từ cache nhanh hơn nhưng không còn augmentation ảnh online, vì ảnh đã được encode cố định
- train từ cache vẫn nặng vì một sample có `8 x 576 = 4608 token`, batch `10` là khoảng `46k token` trước khi qua transformer predictor

Vì sao mặc định dùng `fp32`?

- `fp32` giữ latent cache chính xác hơn `fp16`.
- Mặc định hiện tại dùng `fp32` để loại bỏ nghi ngờ do precision khi đang thử nghiệm model.
- Khi train, Dataset đọc cache và convert latent lại về `float32` trước khi đưa vào predictor.
- Đổi lại, cache `fp32` tốn disk gấp đôi `fp16`.
- Nếu thiếu disk, có thể dùng `--dtype fp16`, nhưng nên xem đó là lựa chọn tiết kiệm dung lượng.
- Train online trước khi trích xuất feature cũng chạy `float32` trong NN-JEPA vì code không bật AMP/autocast.

Dung lượng feature cache với data hiện tại:

```text
manifest frame_count = 49,258
tokens_per_frame = 576
embed_dim = 768
per_frame_fp16 = 0.84375 MiB
per_frame_fp32 = 1.6875 MiB
total_fp16 ~= 40.59 GiB
total_fp32 ~= 81.17 GiB
```

Lệnh extract `fp16` nếu muốn tiết kiệm dung lượng:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --vjepa-checkpoint checkpoints/vjepa2_1/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder vit_base_384 \
  --checkpoint-key ema_encoder \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 32 \
  --dtype fp16
```

Nếu đổi `--dtype`, `--encoder`, `--image-size`, `--patch-size` hoặc `--tubelet-size`, nên đổi `--output-dir`. Script extract cũng kiểm tra shape/dtype của cache cũ; nếu không khớp nó sẽ báo lỗi thay vì âm thầm dùng nhầm feature.

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

Sau lần preprocess gần nhất:

```text
train: 29195 samples
val:    6484 samples
test:   13579 samples
```

Report chi tiết nằm ở:

```text
data/processed/reports/preprocess_report.json
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
src/tools/export_session_gif.py
src/tools/session_web_viewer.py
src/tools/train_rc_car.py
src/tools/train_rc_jepa_ac.py
src/tools/extract_vjepa_features.py
src/tools/train_rc_jepa_ac_features.py
src/tools/eval_rc_jepa_ac_features.py
src/tools/infer_rc_jepa_ac_features.py
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
- split train / val / test
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
python3 -m compileall src tests
```

Chạy test:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Progress Bar Khi Train

`tools.train_rc_car`, `tools.train_rc_jepa_ac` và `tools.train_rc_jepa_ac_features` đều chạy `val` ngay sau mỗi epoch và hiển thị progress bar bằng `tqdm` cho `train`, `val`, `test`.

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

Ngoài metric theo epoch, loop train còn log thêm `train_batch/*` theo batch để nhìn thấy loss curve ngay trong lúc epoch đang chạy.

Trục step trên W&B:

- `train_batch/*` log theo `global_step`.
- `train/*`, `val/*`, `test/*`, `best/*` cũng log theo `global_step`.
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

Nếu muốn chạy không gửi lên W&B:

```bash
--no-wandb
```

Nếu muốn log offline rồi sync sau:

```bash
--wandb-mode offline
```
