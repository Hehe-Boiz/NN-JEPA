# Báo cáo triển khai temporal spacing theo tinh thần V-JEPA2

Ngày viết: 2026-06-09

Phạm vi: source `NN-JEPA` hiện tại, chủ yếu các file `src/data/sequence_dataset.py`, `src/data/feature_sequence_dataset.py`, trainer feature-cache, Hydra configs, eval/infer/planning runtime.

## 1. Vấn đề ban đầu

Trước thay đổi này, một sample train của NN-JEPA được tạo như sau:

```text
raw_frames_per_sample = 8
sequence_stride = 1
sample 1 = [frame 0, frame 1, frame 2, frame 3, frame 4, frame 5, frame 6, frame 7]
sample 2 = [frame 1, frame 2, frame 3, frame 4, frame 5, frame 6, frame 7, frame 8]
```

Tức là `T=8` đang tương đương 8 frame sát nhau trong manifest.

Trong source public V-JEPA2 DROID, `T=8` không nhất thiết là 8 frame raw sát nhau. Source dùng target FPS:

```python
fstp = ceil(video_fps / target_fps)
indices = np.arange(start, start + nframes, fstp)
```

Với `target_fps=4`, nếu video raw khoảng 30 FPS thì sample có thể là:

```text
[100, 108, 116, 124, 132, 140, 148, 156]
```

Không phải:

```text
[100, 101, 102, 103, 104, 105, 106, 107]
```

Ý nghĩa: model học dynamics trên khoảng thời gian dài hơn giữa hai frame kế tiếp trong sample, gần hơn với hành vi robot/xe thay vì chỉ học biến đổi rất nhỏ giữa hai frame sát nhau.

## 2. Kết luận ngắn

Đã triển khai temporal spacing vào code hiện tại.

Có hai cách bật:

1. `frame_stride`: stride cố định bên trong sample.
2. `target_fps`: tự ước lượng FPS của từng session từ timestamp, rồi dùng `ceil(source_fps / target_fps)` giống tinh thần source Meta.

Default vẫn giữ:

```yaml
frame_stride: 1
target_fps: 0.0
```

Nghĩa là nếu không bật config mới thì behavior cũ không đổi.

## 3. Thay đổi code đã làm

### 3.1. `src/data/settings.py`

Thêm global config:

```python
AC_FRAME_STRIDE = 1
AC_TARGET_FPS = 0.0
```

Ý nghĩa:

- `AC_FRAME_STRIDE`: khoảng cách frame cố định bên trong một sample.
- `AC_TARGET_FPS`: nếu > 0, dataset tự ước lượng FPS từng session và chọn stride theo target FPS.

Thứ tự ưu tiên:

```text
Nếu target_fps > 0:
    dùng target_fps để tính effective_frame_stride
Nếu target_fps == 0:
    dùng frame_stride
Nếu target_fps > 0 nhưng không ước lượng được FPS:
    fallback về frame_stride
```

### 3.2. `src/data/sequence_dataset.py`

Đây là thay đổi lõi.

Các tham số mới được thêm vào:

```python
frame_stride: int = settings.AC_FRAME_STRIDE
target_fps: float = settings.AC_TARGET_FPS
```

Áp dụng cho:

- `RCJepaACSequenceDataset`
- `build_sequence_windows`
- `create_ac_sequence_dataloaders`

Hàm mới:

```python
resolve_effective_frame_stride(samples, indices, frame_stride, target_fps)
```

Logic:

```text
Nếu target_fps <= 0:
    effective_frame_stride = frame_stride
Nếu target_fps > 0:
    source_fps = estimate_source_fps(session)
    effective_frame_stride = ceil(source_fps / target_fps)
Nếu không estimate được source_fps:
    effective_frame_stride = frame_stride
```

Hàm mới:

```python
estimate_source_fps(samples, indices)
```

Logic:

1. Duyệt các frame liên tiếp trong session.
2. Lấy `timestamp_sec`.
3. Tính `time_gap / frame_gap`.
4. Lấy median frame period.
5. FPS = `1 / median_period`.

Vì dùng median nên ít bị ảnh hưởng bởi một vài timestamp outlier.

Hàm mới:

```python
median_float(values)
```

Dùng để tính median frame period.

### 3.3. `build_sequence_windows` sau thay đổi

Trước đây:

```python
window = indices[start : start + raw_frames_per_sample]
```

Sau thay đổi:

```python
window = [
    indices[start + (offset * effective_frame_stride)]
    for offset in range(raw_frames_per_sample)
]
```

Ví dụ:

```text
raw_frames_per_sample = 8
frame_stride = 2
sample = [0, 2, 4, 6, 8, 10, 12, 14]
```

Ví dụ target FPS:

```text
source_fps = 10
target_fps = 4
effective_frame_stride = ceil(10 / 4) = 3
sample = [0, 3, 6, 9, 12, 15, 18, 21]
```

### 3.4. `is_contiguous_window` sau thay đổi

Trước đây, window chỉ hợp lệ nếu frame gap gần như 1:

```text
frame 0 -> frame 1 -> frame 2
```

Sau thay đổi, window hợp lệ theo expected stride:

```text
frame_stride = 2:
    frame 0 -> frame 2 -> frame 4

target_fps sinh effective_frame_stride = 3:
    frame 0 -> frame 3 -> frame 6
```

`max_time_gap_sec` cũng được scale theo stride:

```python
max_allowed_time_gap = max_time_gap_sec * expected_frame_stride
```

Lý do:

- Nếu chọn frame cách nhau 2 hoặc 3 frame raw, khoảng thời gian thật giữa hai frame trong sample phải được phép lớn hơn.
- Nếu không scale, dataset sẽ tự reject hầu hết temporal-spaced samples.

### 3.5. `src/data/feature_sequence_dataset.py`

Pipeline train chính của bạn đang dùng feature cache, nên file này cũng được cập nhật.

Thêm tham số:

```python
frame_stride
target_fps
```

Truyền xuống:

```python
build_sequence_windows(...)
```

Điều này rất quan trọng: không cần extract feature lại chỉ để đổi temporal spacing, vì feature cache lưu feature từng frame. Dataset chỉ chọn frame nào để ghép thành sample.

### 3.6. `src/tools/train_rc_jepa_ac_features.py`

Thêm CLI args:

```bash
--frame-stride
--target-fps
```

Trainer truyền hai tham số này vào dataloader.

Checkpoint mới sẽ lưu hai field này trong `args`, vì `save_checkpoint` đã lưu toàn bộ args.

Resume validation cũng được cập nhật:

```python
checked_fields = (
    "predictor_type",
    "raw_frames_per_sample",
    "frame_stride",
    "target_fps",
    "auto_steps",
    "predictor_dim",
    "predictor_depth",
    "predictor_heads",
    "dropout",
)
```

Ý nghĩa:

- Nếu checkpoint cũ train với `frame_stride=1` mà bạn resume bằng config `frame_stride=2`, code sẽ báo lỗi.
- Đây là cố ý để tránh vô tình trộn hai experiment khác nhau.
- Với checkpoint rất cũ chưa có field `frame_stride`/`target_fps`, code xem nó như baseline `frame_stride=1`, `target_fps=0.0`, rồi vẫn so với config hiện tại. Vì vậy checkpoint cũ cũng không thể bị resume nhầm sang temporal experiment.

### 3.7. `src/tools/train_rc_jepa_ac.py`

Trainer online/raw-image cũng được cập nhật cùng tham số:

```bash
--frame-stride
--target-fps
```

Dù hiện tại bạn train chủ yếu từ feature cache, việc này giữ hai pipeline đồng bộ.

### 3.8. `src/tools/train_rc_jepa_ac_features_hydra.py`

Hydra bridge được cập nhật:

```python
args.frame_stride = int(data_cfg.get("frame_stride", args.frame_stride))
args.target_fps = float(data_cfg.get("target_fps", args.target_fps))
```

Nghĩa là YAML có thể điều khiển temporal spacing trực tiếp.

### 3.9. Runtime eval/infer/planning

Các file được cập nhật:

- `src/tools/rc_jepa_ac_feature_runtime.py`
- `src/tools/eval_rc_jepa_ac_features.py`
- `src/tools/infer_rc_jepa_ac_features.py`
- `src/tools/plan_rc_jepa_ac_features.py`

`FeaturePredictorConfig` có thêm:

```python
frame_stride: int
target_fps: float
```

Nếu checkpoint cũ không có hai field này:

```python
frame_stride = settings.AC_FRAME_STRIDE  # 1
target_fps = settings.AC_TARGET_FPS      # 0.0
```

Vì vậy checkpoint cũ vẫn eval/infer được.

Eval/infer/planning sẽ rebuild dataloader đúng temporal spacing của checkpoint mới.

## 4. Config Hydra mới

Đã thêm hai experiment mới.

### 4.1. `rc_jepa_tiny_temporal_fps4_newdata`

File:

```text
configs/hydra/experiment/rc_jepa_tiny_temporal_fps4_newdata.yaml
```

Thông số chính:

```yaml
data:
  raw_frames_per_sample: 8
  sequence_stride: 1
  frame_stride: 1
  target_fps: 4.0
  auto_steps: 2

model:
  type: simple
  size: tiny
```

Ý nghĩa:

- Gần source Meta hơn vì tính stride từ FPS từng session.
- Nếu session khoảng 8.2 FPS, `ceil(8.2 / 4) = 3`.
- Effective FPS thực tế khoảng `8.2 / 3 = 2.7 FPS`.
- Đây là hệ quả của dùng `ceil` giống source; với FPS thấp, stride bị nhảy thô.

Lệnh chạy:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_temporal_fps4_newdata
```

### 4.2. `rc_jepa_tiny_frame_stride2_newdata`

File:

```text
configs/hydra/experiment/rc_jepa_tiny_frame_stride2_newdata.yaml
```

Thông số chính:

```yaml
data:
  raw_frames_per_sample: 8
  sequence_stride: 1
  frame_stride: 2
  target_fps: 0.0
  auto_steps: 2

model:
  type: simple
  size: tiny
```

Ý nghĩa:

- Không tự ước lượng FPS.
- Luôn lấy cách 2 frame:

```text
[0,2,4,6,8,10,12,14]
```

- Với data hiện tại khoảng 8.2 FPS, effective FPS khoảng `8.2 / 2 = 4.1 FPS`.
- Đây là hướng thực dụng gần repo JEPA của bạn bạn hơn.

Lệnh chạy:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_frame_stride2_newdata
```

## 5. Số liệu kiểm tra trên manifest hiện tại

Tôi đã thống kê trực tiếp trên `data/processed/manifests`.

Kết quả:

| Split | Samples | Sessions | Window stride1 | Window frame_stride2 | Window target_fps4 | Avg FPS |
|---|---:|---:|---:|---:|---:|---:|
| train | 84.766 | 70 | 68.139 | 59.173 | 53.896 | 8.227 |
| val | 9.470 | 15 | 8.259 | 7.462 | 7.114 | 8.273 |
| test | 20.310 | 15 | 16.138 | 13.642 | 12.458 | 8.056 |

Nhận xét:

- `frame_stride=2` giảm số train windows khoảng 13%.
- `target_fps=4` giảm số train windows khoảng 21%.
- Vì FPS hiện tại chỉ khoảng 8.2, `target_fps=4` theo `ceil` sẽ thường chọn stride 3, nên temporal gap lớn hơn khá nhiều.
- Nếu mục tiêu là gần 4 FPS thật, `frame_stride=2` hợp lý hơn trên data hiện tại.
- Nếu mục tiêu là bám sát logic source Meta, `target_fps=4` hợp lý hơn.

## 6. Có cần train lại từ đầu không?

Câu trả lời thực dụng: có, nên train lại từ đầu cho experiment temporal spacing.

Lý do:

- Temporal spacing làm thay đổi distribution của sample.
- Model cũ học chuyển động từ frame sát nhau:

```text
z_t -> z_{t+1} với delta thời gian nhỏ
```

- Model mới học chuyển động xa hơn:

```text
z_t -> z_{t+k} với k = 2 hoặc 3
```

Về mặt shape tensor:

- `raw_frames_per_sample` vẫn là 8.
- `tokens_per_frame` vẫn là 576.
- `embed_dim` vẫn là 768.
- Predictor architecture không đổi nếu vẫn dùng cùng model size.

Nghĩa là về kỹ thuật có thể lấy weight cũ để fine-tune. Nhưng code hiện tại đã cố tình chặn resume nếu `frame_stride/target_fps` khác checkpoint, vì resume như vậy dễ làm bạn tưởng là cùng một run, trong khi thực chất là một experiment khác.

Kết luận:

- Nếu muốn so sánh nghiêm túc: train fresh, không resume checkpoint cũ.
- Nếu muốn tận dụng weight cũ để fine-tune: nên thêm cơ chế riêng kiểu `--init-from`, không nên dùng `--resume-from`. Hiện chưa triển khai `--init-from`.

## 7. Có cần extract feature lại không?

Nếu chỉ đổi temporal spacing:

```text
Không cần extract feature lại.
```

Lý do:

- Feature cache lưu feature từng frame.
- Temporal spacing chỉ đổi cách chọn frame thành một sample.
- Ví dụ cache đã có frame 0,1,2,3,4,5,6,7,8.
- Dataset cũ chọn `[0,1,2,3,4,5,6,7]`.
- Dataset mới chọn `[0,2,4,6,8,10,12,14]`.
- Miễn feature của các frame đó đã có trong cache thì không cần chạy encoder lại.

Nếu thêm data servo cũ vào `data/raw` rồi preprocess lại:

```text
Cần chạy feature extractor lại, nhưng không nhất thiết extract lại toàn bộ.
```

Lý do:

- Extractor hiện có logic cache.
- Session đã có `.npy/.json` hợp lệ sẽ được skip.
- Session mới chưa có feature sẽ được extract thêm.

Lệnh extract lại an toàn:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --batch-size 32 \
  --dtype fp32
```

## 8. Có nên lấy cả data servo cũ để train không?

Câu trả lời: nên thử, nhưng phải coi là experiment riêng, không nên trộn vào baseline chính rồi kết luận vội.

### 8.1. Vì sao có thể có lợi

Data servo cũ có thể giúp:

- Tăng số lượng trajectory.
- Tăng đa dạng cảnh/ánh sáng/sàn nhà/góc camera.
- Giúp encoder/predictor thấy nhiều tình huống chuyển động hơn.
- Giảm overfit vào vài session mới.

Với world model latent, nhiều dữ liệu thường có lợi nếu action/state semantics nhất quán.

### 8.2. Vì sao có thể gây hại

Servo cũ có thể gây domain shift:

- Cùng `steering_cmd_t = 0.3` nhưng góc lái thật khác servo mới.
- Latency servo cũ khác.
- Deadzone servo cũ khác.
- Tốc độ phản hồi khác.
- Throttle/steering calibration khác.
- Cơ khí bánh xe/càng lái có thể khác.

Vì predictor học:

```text
latent_t + state_t + action_t -> latent_{t+1}
```

nên nếu action semantics khác, model có thể bị nhiễu dynamics.

Ví dụ xấu:

```text
Data mới:
  steering_cmd_t = 0.4 -> xe rẽ mạnh

Data servo cũ:
  steering_cmd_t = 0.4 -> xe rẽ nhẹ
```

Nếu trộn không có domain label, model phải học trung bình hai dynamics này. Kết quả closed-loop có thể tệ.

### 8.3. Nên thử theo thứ tự nào

Tôi đề xuất thứ tự experiment:

1. New data only + `frame_stride=2` + tiny.
2. New data only + `target_fps=4` + tiny.
3. New + old servo mix + `frame_stride=2` + tiny.
4. Nếu tiny ổn, chạy base với cấu hình thắng.
5. Nếu mix old servo tốt hơn, cân nhắc thêm `servo_type/domain_id` vào state sau này.

Không nên nhảy thẳng vào base + full old/new mix vì mỗi epoch lâu, khó debug nguyên nhân.

### 8.4. Khuyến nghị cụ thể hiện tại

Với số liệu hiện tại FPS khoảng 8.2:

- Chạy `frame_stride=2` trước.
- Sau đó chạy `target_fps=4` để so với source-style.
- Nếu cả hai đều ổn, mới thử thêm servo cũ.

Lý do:

- `frame_stride=2` cho effective FPS khoảng 4.1, gần mục tiêu 4 FPS hơn.
- `target_fps=4` bám source hơn nhưng do dùng `ceil`, với data 8.2 FPS sẽ thành stride 3, effective FPS khoảng 2.7.

## 9. Lệnh train đề xuất

### 9.1. Thử nhanh hướng thực dụng `frame_stride=2`

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_frame_stride2_newdata
```

### 9.2. Thử hướng source-style `target_fps=4`

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_temporal_fps4_newdata
```

### 9.3. Nếu muốn chạy base sau khi tiny ổn

Có thể override trực tiếp:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_frame_stride2_newdata \
  model.size=base \
  train.batch_size=10 \
  train.eval_batch_size=2 \
  train.warmup_epochs=4 \
  output_dir=checkpoints/rc_jepa_ac_vitb_features_newdata_base_frame_stride2 \
  wandb.run_name=rc-jepa-base-newdata-frame-stride2
```

Với source-style:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_temporal_fps4_newdata \
  model.size=base \
  train.batch_size=10 \
  train.eval_batch_size=2 \
  train.warmup_epochs=4 \
  output_dir=checkpoints/rc_jepa_ac_vitb_features_newdata_base_temporal_fps4 \
  wandb.run_name=rc-jepa-base-newdata-temporal-fps4
```

## 10. Kiểm tra đã chạy

### 10.1. Compile source

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python3 -m py_compile src/data/*.py src/models/*.py src/tools/*.py
```

Kết quả: không có lỗi compile.

### 10.2. Test synthetic window

Với data giả 10 FPS:

```text
frame_stride=2 -> [0, 2, 4, 6]
target_fps=4  -> [0, 3, 6, 9]
```

Kết quả đúng với thiết kế:

- fixed stride chọn đúng stride 2.
- target FPS dùng `ceil(10 / 4) = 3`.

### 10.3. Hydra dry-run

Đã chạy dry-run cho:

```bash
experiment=rc_jepa_tiny_temporal_fps4_newdata
experiment=rc_jepa_tiny_frame_stride2_newdata
```

Kết quả:

- Config `target_fps=4.0` được truyền đúng vào trainer.
- Config `frame_stride=2` được truyền đúng vào trainer.
- Output dirs riêng.
- W&B run names riêng.

### 10.4. Resume guard và forward loss

Đã test thêm:

- Checkpoint giả lập kiểu cũ không có `frame_stride` bị chặn khi resume bằng `frame_stride=2`.
- Dataloader thật với `frame_stride=2` tạo batch shape:

```text
latents = [1, 4608, 768]
states  = [1, 8, 5]
actions = [1, 7, 2]
```

- Predictor tiny chạy được `compute_world_model_losses` trên batch thật và trả đủ:

```text
loss
teacher_forcing_loss
rollout_loss
```

## 11. Rủi ro còn lại

### 11.1. `target_fps=4` với FPS thấp có thể undersample

Vì source dùng `ceil`, nếu data chỉ 8.2 FPS:

```text
ceil(8.2 / 4) = 3
effective FPS = 8.2 / 3 = 2.7
```

Nó bám source logic nhưng không bám đúng target FPS theo nghĩa toán học. Đây là lý do tôi thêm cả config `frame_stride=2`.

### 11.2. Mixing old servo data có thể cần domain conditioning

Nếu old servo khác dynamics nhiều, hướng tốt hơn về lâu dài là thêm một input:

```text
servo_type / domain_id / hardware_id
```

vào state hoặc embedding riêng. Hiện chưa làm để giữ pipeline đơn giản.

### 11.3. Feature cache vẫn là fp32 token-level

Temporal spacing không làm feature nhẹ hơn.

Mỗi sample vẫn có:

```text
T = 8
tokens_per_frame = 576
embed_dim = 768
latents = [4608, 768]
```

Chỉ khác là 8 frame này cách xa nhau hơn theo thời gian.

## 12. Kết luận

Temporal spacing đã được triển khai đúng hướng và không phá behavior cũ. Default vẫn là contiguous frame. Muốn bật thì dùng Hydra experiment mới hoặc CLI args mới.

Khuyến nghị chạy trước:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_frame_stride2_newdata
```

Sau đó chạy đối chứng source-style:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_temporal_fps4_newdata
```

Không nên resume checkpoint cũ cho experiment này. Nên train fresh để so sánh sạch. Nếu chỉ đổi temporal spacing thì không cần extract feature lại. Nếu thêm data servo cũ thì cần preprocess lại và chạy feature extractor lại để extract các session mới còn thiếu.
