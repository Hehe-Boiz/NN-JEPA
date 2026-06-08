# Báo cáo: V-JEPA 2-AC official so với NN-JEPA hiện tại

Ngày viết: 2026-06-09

Repo đang xét:

- `vjepa2/`: repo public V-JEPA 2/V-JEPA 2.1 được clone về để tham khảo encoder, checkpoint và pipeline action-conditioned.
- `NN-JEPA`: repo train hiện tại cho xe RC, dùng V-JEPA 2.1 làm frozen encoder và train predictor/world model riêng.

## Kết luận ngắn

V-JEPA 2-AC official trong repo public không mean-pool latent khi train action-conditioned world model. Pipeline official dùng latent dạng token-level, tức mỗi frame vẫn giữ toàn bộ patch tokens, sau đó predictor nhận latent tokens + action tokens + state tokens và dự đoán latent tokens của frame tương lai.

NN-JEPA hiện tại đang đi đúng hướng ở mức ý tưởng lớn:

- Dùng frozen V-JEPA 2.1 encoder để lấy feature.
- Lưu feature dạng full token layout `[N, K, D]`, không phải pooled vector `[N, D]`.
- Predictor nhận latent tokens + state + action.
- Loss gồm `teacher_forcing_loss` và `rollout_loss`.
- `auto_steps = 2`.
- Khi train từ feature cache thì encoder không được train lại.

Nhưng NN-JEPA hiện tại chưa phải bản official exact của V-JEPA 2-AC:

- Predictor đang là `SimpleACPredictor`, đơn giản hơn `VisionTransformerPredictorAC` official.
- Attention mask đang là time-causal đơn giản, chưa phải action-block causal attention mask official.
- State/action của xe RC khác DROID robot: NN-JEPA dùng state 5D và action 2D, official DROID dùng state/action robot 7D.
- NN-JEPA đang dùng V-JEPA 2.1 ViT-B/16 384 làm encoder chính, còn checkpoint AC official public là V-JEPA 2-AC từ ViT-g/16 256.
- NN-JEPA đang dùng `fp32` để giữ độ chính xác feature cache, còn config official dùng `bfloat16`.

Vì vậy cách gọi chính xác nhất hiện tại là:

> NN-JEPA = V-JEPA 2.1 frozen encoder + custom action-conditioned world-model predictor cho xe RC, bám tinh thần V-JEPA-AC nhưng chưa phải implementation official exact.

## Điểm cần chỉnh chính xác về số token

Khi nói "token-level latent", cần phân biệt resolution:

- Official V-JEPA 2-AC public dùng `crop_size = 256`, `patch_size = 16`, nên mỗi frame có `16 x 16 = 256` patch tokens.
- NN-JEPA hiện tại dùng V-JEPA 2.1 `384 x 384`, `patch_size = 16`, nên mỗi frame có `24 x 24 = 576` patch tokens.

Vì vậy:

- Official public AC: latent theo thời gian có dạng gần đúng `[B, T * 256, D]`.
- NN-JEPA ViT-B 384 hiện tại: latent có dạng `[B, T * 576, 768]`.

Ý chính vẫn giống nhau: không mean-pool về một vector duy nhất cho mỗi frame.

## Bằng chứng từ repo `vjepa2`

### 1. README public tách rõ V-JEPA 2, V-JEPA 2.1 và V-JEPA 2-AC

Trong `vjepa2/README.md`, repo public liệt kê:

- V-JEPA 2 pretrained checkpoints:
  - ViT-L/16 256.
  - ViT-H/16 256.
  - ViT-g/16 256.
  - ViT-g/16 384.

- V-JEPA 2.1 pretrained checkpoints:
  - ViT-B/16 384, khoảng 80M params.
  - ViT-L/16 384, khoảng 300M params.
  - ViT-g/16 384, khoảng 1B params.
  - ViT-G/16 384, khoảng 2B params.

- V-JEPA 2-AC:
  - Checkpoint action-conditioned public là `vjepa2-ac-vitg.pt`.
  - README ghi rõ action-conditioned checkpoint được train từ encoder ViT-g.
  - Config training được trỏ tới `configs/train/vitg16/droid-256px-8f.yaml`.
  - PyTorch Hub example là `torch.hub.load('facebookresearch/vjepa2', 'vjepa2_ac_vit_giant')`.

Điểm quan trọng: trong source public hiện tại, mình không thấy một config/checkpoint action-conditioned riêng, sạch, tên kiểu "V-JEPA 2.1-AC". Phần 2.1 public chủ yếu là pretrained encoder/backbone 384. Phần AC public là V-JEPA 2-AC từ ViT-g/16 256.

### 2. Config official DROID AC

File official đã kiểm tra:

`vjepa2/configs/train/vitg16/droid-256px-8f.yaml`

Các tham số quan trọng:

```yaml
app: vjepa_droid
data:
  batch_size: 8
  crop_size: 256
  dataset_fpcs:
  - 8
  fps: 4
  patch_size: 16
  tubelet_size: 2
loss:
  auto_steps: 2
  loss_exp: 1.0
  normalize_reps: true
meta:
  dtype: bfloat16
  pretrain_checkpoint: /your_vjepa2_checkpoints/vitg.pt
  context_encoder_key: target_encoder
  target_encoder_key: target_encoder
model:
  model_name: vit_giant_xformers
  pred_depth: 24
  pred_embed_dim: 1024
  pred_is_frame_causal: true
  pred_num_heads: 16
  use_activation_checkpointing: true
  use_extrinsics: false
  use_rope: true
optimization:
  epochs: 315
  ipe: 300
  lr: 0.000425
  start_lr: 0.000075
  warmup: 15
  final_lr: 0.0
  weight_decay: 0.04
```

Ý nghĩa:

- Mỗi sample official có 8 frame video, vì `dataset_fpcs = 8`.
- Ảnh train official là 256, vì `crop_size = 256`.
- Patch size là 16, nên mỗi frame là 256 tokens.
- `tubelet_size = 2`, nhưng train loop official encode từng frame bằng pseudo-clip có 2 frame lặp lại, chứ không phải encode trực tiếp cả 8 frame thành 4 tubelet temporal tokens theo cách dễ nhầm.
- Predictor official lớn: depth 24, predictor dim 1024, 16 heads.
- Official dùng `bfloat16`, activation checkpointing và multi-GPU lớn. Config ghi `nodes = 4`, `tasks_per_node = 8`, `mem_per_gpu = 220G`, tức không phải setup nhẹ.

### 3. Train loop official dùng target encoder và predictor

File đã kiểm tra:

`vjepa2/app/vjepa_droid/train.py`

Trong training setup:

- Code tạo `encoder`, `predictor`, `target_encoder`.
- `target_encoder` là copy của `encoder`.
- `target_encoder` bị freeze bằng `requires_grad = False`.
- Checkpoint pretrained được load vào encoder/target encoder qua key trong config.

Trong forward target official, code xử lý clip như sau về mặt logic:

```python
c = c.permute(0, 2, 1, 3, 4)
c = c.flatten(0, 1)
c = c.unsqueeze(2).repeat(1, 1, 2, 1, 1)
h = target_encoder(c)
h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
```

Giải thích shape:

- Input `clips` ban đầu là `[B, C, T, H, W]`.
- Code đổi thành từng frame riêng: `[B*T, C, H, W]`.
- Mỗi frame được biến thành pseudo-clip 2 frame giống nhau: `[B*T, C, 2, H, W]`.
- `target_encoder` encode pseudo-clip này.
- Kết quả được reshape lại thành `[B, T, K, D]`, rồi flatten thành `[B, T*K, D]`.

Vì official dùng `crop_size = 256`, `patch_size = 16`, nên `K = 256`. Nếu dùng ảnh 384 như NN-JEPA thì `K = 576`.

Điểm quan trọng: không có bước mean-pool token trong train loop official. `h` vẫn là token-level latent.

### 4. Loss official gồm teacher forcing và autoregressive rollout

Trong `vjepa2/app/vjepa_droid/train.py`, phần prediction có hai nhánh chính:

1. Teacher forcing:

```python
_z = z[:, :-tokens_per_frame]
_a = actions
_s = states[:, :-1]
z_tf = predictor(_z, _a, _s, _e)
```

Ý nghĩa:

- Đưa latent thật của các frame trước vào predictor.
- Dùng action/state tương ứng.
- Dự đoán latent frame tiếp theo.

2. Autoregressive rollout:

```python
_z = torch.cat([z[:, :tokens_per_frame], z_tf[:, :tokens_per_frame]], dim=1)
for n in range(1, auto_steps):
    _z_nxt = predictor(_z, actions[:, :n+1], states[:, :n+1], extrinsics[:, :n+1])[:, -tokens_per_frame:]
    _z = torch.cat([_z, _z_nxt], dim=1)
```

Ý nghĩa:

- Bắt đầu từ latent frame đầu.
- Dùng output đã dự đoán làm input cho bước tiếp theo.
- Với `auto_steps = 2`, model bị ép học không chỉ một bước bằng teacher forcing, mà còn học rollout ngắn qua prediction của chính nó.

Loss official:

```python
jloss = loss_fn(z_tf, h)
sloss = loss_fn(z_ar, h)
loss = jloss + sloss
```

Trong config, `loss_exp = 1.0`, nên loss là dạng L1 trung bình:

```python
mean(abs(pred - target))
```

### 5. Predictor official là token-level action-conditioned transformer

File đã kiểm tra:

`vjepa2/src/models/ac_predictor.py`

Class official:

```python
VisionTransformerPredictorAC
```

Nó nhận:

- `x`: latent tokens.
- `actions`: action tokens.
- `states`: state tokens.
- `extrinsics`: optional.

Các bước chính:

1. Project latent từ encoder dim sang predictor dim:

```python
x = self.predictor_embed(x)
```

2. Encode action/state thành token:

```python
s = self.state_encoder(states).unsqueeze(2)
a = self.action_encoder(actions).unsqueeze(2)
```

3. Reshape latent về frame structure:

```python
x = x.view(B, T, H*W, D)
```

4. Ghép action token, state token và patch tokens theo từng frame:

```python
x = torch.cat([a, s, x], dim=2).flatten(1, 2)
```

5. Chạy qua các `ACBlock` với action-block causal attention mask:

```python
attn_mask = build_action_block_causal_attention_mask(...)
```

6. Bỏ action/state token ra, chỉ giữ lại predicted latent tokens:

```python
x = x.view(B, T, cond_tokens + H*W, D)
x = x[:, :, cond_tokens:, :].flatten(1, 2)
x = self.predictor_proj(self.predictor_norm(x))
```

Điểm quan trọng: output vẫn là latent tokens, không phải action trực tiếp, không phải pooled vector.

## NN-JEPA hiện tại đang làm gì

### 1. Encoder: frozen V-JEPA 2.1

File chính:

`src/models/rc_jepa_ac.py`

Class:

```python
FrozenVJepa21Encoder
```

Ý tưởng:

- Build encoder từ source `vjepa2/app/vjepa_2_1/models/vision_transformer.py`.
- Load checkpoint V-JEPA 2.1.
- Freeze toàn bộ parameter encoder.
- Set encoder ở eval mode.
- Dùng encoder như target feature extractor, không train encoder trong phase train predictor từ feature cache.

Forward của NN-JEPA cũng encode frame độc lập thành pseudo-clip:

```python
frames = images.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)
pseudo_clips = frames.unsqueeze(2).repeat(1, 1, tubelet_size, 1, 1)
tokens = self.encoder(pseudo_clips)
tokens = F.layer_norm(tokens, (tokens.size(-1),))
tokens = tokens.view(B, T, tokens_per_frame, D)
return tokens.flatten(1, 2)
```

Cách này rất gần với target encoding trong train loop official DROID AC. Khác biệt lớn là NN-JEPA dùng V-JEPA 2.1 encoder 384 thay vì V-JEPA 2 ViT-g 256.

### 2. Feature cache: full-token, không pooled

File chính:

`src/tools/extract_vjepa_features.py`

Metadata hiện tại ghi:

- `feature_layout = "frame_tokens"`
- `image_size = 384`
- `patch_size = 16`
- `tubelet_size = 2`
- `tokens_per_frame = 576`
- `embed_dim = 768` nếu dùng ViT-B 384.
- `dtype = fp32` nếu dùng lệnh hiện tại.

Mỗi session được lưu thành:

- `.npy`: feature array.
- `.json`: index map từ `frame_index` sang row trong `.npy`.

Shape kỳ vọng trong extractor:

```python
(len(session_samples), tokens_per_frame, embed_dim)
```

Với ViT-B 384:

```text
[số frame trong session, 576, 768]
```

Do đó cache hiện tại là full-token cache. Đây là hướng giống AC official hơn pooled cache.

### 3. DataLoader feature bắt buộc đọc full-token

File chính:

`src/data/feature_sequence_dataset.py`

Dataset:

```python
RCJepaACFeatureSequenceDataset
```

Nó load feature session bằng:

```python
feature_array = np.load(npy_path, mmap_mode="r")
if feature_array.ndim != 3:
    raise ValueError(...)
```

Tức là DataLoader hiện tại yêu cầu feature có shape 3D `[N, K, D]`. Nếu dùng pooled feature `[N, D]`, code hiện tại sẽ báo lỗi và không silently train sai.

Mỗi sample train hiện tại:

- `raw_frames_per_sample = 8`.
- `sequence_stride = 1`.
- Latents: 8 frame, mỗi frame 576 tokens nếu ViT-B 384.
- Shape latent trong một sample: `[8 * 576, 768] = [4608, 768]`.
- States: `[8, 5]`.
- Actions: `[7, 2]`.

State columns hiện tại:

```text
yaw_rate_t
accel_x_t
accel_y_t
steering_last_t
throttle_last_t
```

Action columns hiện tại:

```text
steering_cmd_t
throttle_cmd_t
```

Điểm quan trọng: action có độ dài `T - 1`, vì action giữa frame `t` và `t+1` dùng để dự đoán latent frame tiếp theo.

### 4. Predictor NN-JEPA hiện tại

File chính:

`src/models/rc_jepa_ac.py`

Class:

```python
SimpleACPredictor
```

Nó làm các bước:

1. Project latent tokens:

```python
self.latent_proj = nn.Linear(latent_dim, predictor_dim)
```

2. Project state/action:

```python
self.state_proj = nn.Linear(state_dim, predictor_dim)
self.action_proj = nn.Linear(action_dim, predictor_dim)
```

3. Thêm frame position, patch position, action type, state type.

4. Ghép action token, state token và latent tokens theo từng frame:

```python
sequence = torch.cat([action, state, latent], dim=2).flatten(1, 2)
```

5. Dùng time-causal mask:

```python
mask = build_time_causal_mask(...)
```

6. Chạy qua `nn.TransformerEncoder`.

7. Bỏ action/state token, project output về latent dim:

```python
predicted = sequence[:, :, self.cond_tokens:, :].flatten(1, 2)
predicted = self.output_proj(self.norm(predicted))
```

Điểm giống official:

- Predictor dự đoán latent tokens, không dự đoán action trực tiếp.
- Action/state được đưa vào như conditioning tokens.
- Output giữ token-level structure.

Điểm khác official:

- Official dùng `ACBlock` riêng, RoPE, action-block causal attention mask.
- NN-JEPA dùng `nn.TransformerEncoderLayer` và time-causal mask đơn giản.
- Official predictor lớn hơn rất nhiều: depth 24, dim 1024, heads 16.
- NN-JEPA có preset tiny/small/base để thử nghiệm nhanh.

### 5. Loss NN-JEPA hiện tại

File chính:

`src/models/rc_jepa_ac.py`

Function:

```python
compute_world_model_losses
```

Loss gồm:

- `teacher_forcing_loss`
- `rollout_loss`
- `loss = teacher_forcing_loss + rollout_loss`

Teacher forcing:

```python
input_latents = latents[:, :-tokens_per_frame]
target_latents = latents[:, tokens_per_frame:]
teacher_pred = predictor(
    latent_tokens=input_latents,
    actions=actions,
    states=states[:, :-1],
)
teacher_forcing_loss = F.l1_loss(teacher_pred, target_latents)
```

Autoregressive rollout:

```python
rollout_steps = min(auto_steps, num_frames - 1)
rollout_tokens = latents[:, :tokens_per_frame]
for step in range(rollout_steps):
    pred_tokens = predictor(...)
    next_tokens = pred_tokens[:, -tokens_per_frame:]
    rollout_tokens = torch.cat([rollout_tokens, next_tokens], dim=1)
rollout_loss = F.l1_loss(rollout_pred, rollout_target)
```

Điểm giống official:

- Có teacher forcing.
- Có rollout autoregressive.
- Loss dùng L1.
- `auto_steps = 2`.

Điểm khác cần chú ý:

- Official dùng measured states trong rollout public loop: `states[:, :n+1]`.
- NN-JEPA tránh đọc future measured state trong rollout bằng cách build state context từ initial state và action, chủ yếu copy action trước đó vào các state field như `steering_last_t`, `throttle_last_t` nếu có mapping.
- Đây là lựa chọn hợp lý hơn cho xe tự hành, vì inference thật không có future measured state. Nhưng nó khác code public official.

## Vì sao pooled latent trong repo `JEPA` không phải chuẩn official AC

Trong repo `JEPA` của bạn, hướng feature cache cũ có mean-pool token:

```python
tok.float().mean(1).cpu()
```

Về ý nghĩa:

- Input token-level: `[B, K, D]`.
- Mean-pool theo chiều token: `[B, D]`.
- Mỗi frame chỉ còn một vector global.

Ưu điểm:

- Cache nhỏ hơn rất nhiều.
- Train nhanh hơn.
- Predictor nhẹ hơn.
- Dễ debug, dễ thử nghiệm nhiều cấu hình.

Nhược điểm:

- Mất bố cục không gian của ảnh.
- Predictor không còn học dynamics trên patch tokens.
- Không giống `VisionTransformerPredictorAC` official.
- Khó mô phỏng đúng cách official ghép action/state token với patch token theo từng frame.

Kết luận thực dụng:

- Pooled latent nên được xem là baseline/ablation nhanh.
- Full-token latent nên là nhánh chính nếu mục tiêu là bám V-JEPA-AC official.

## So sánh trực tiếp official AC và NN-JEPA

| Thành phần | V-JEPA 2-AC official public | NN-JEPA hiện tại |
|---|---|---|
| Encoder | V-JEPA 2 ViT-g/16 | V-JEPA 2.1 ViT-B/16 384 mặc định |
| Checkpoint AC public | `vjepa2-ac-vitg.pt` | Chưa dùng checkpoint AC official, train custom predictor |
| Resolution | 256 | 384 |
| Patch size | 16 | 16 |
| Token/frame | 256 | 576 |
| Frame/sample | 8 | 8 |
| Tubelet size | 2 | 2 |
| Feature layout | Token-level | Token-level |
| Mean-pool | Không | Không |
| State/action | DROID 7D/7D | RC 5D/2D |
| Predictor | `VisionTransformerPredictorAC` | `SimpleACPredictor` |
| Predictor depth | 24 | tiny 2, small 4, base 6 |
| Predictor dim | 1024 | tiny 128, small 256, base 512 |
| Attention | ACBlock + RoPE + action-block causal mask | TransformerEncoder + time-causal mask |
| Loss | teacher forcing + rollout | teacher forcing + rollout |
| `auto_steps` | 2 | 2 |
| Precision | bfloat16 | fp32 feature cache hiện tại |

## Tác động thực tế tới train xe RC

### 1. Full-token đúng hướng hơn nhưng rất nặng

Với ViT-B 384:

```text
1 frame = 576 tokens x 768 dim x 4 bytes fp32
        ~= 1.77 MB/frame
```

Với 8 frame/sample:

```text
1 sample latent ~= 8 x 1.77 MB = 14.16 MB
```

Đây chỉ là input latent, chưa tính activation của predictor. Vì vậy full-token train rất dễ OOM, đặc biệt ở validation nếu batch size lớn. Việc NN-JEPA hạ `eval_batch_size` về 2 là hợp lý.

### 2. ViT-L/g/G mạnh hơn nhưng cache và VRAM tăng rất mạnh

Nếu giữ full-token fp32:

- ViT-B 384: `K = 576`, `D = 768`.
- ViT-L 384: `K = 576`, `D = 1024`.
- ViT-g 384: `K = 576`, `D = 1408`.
- ViT-G 384: `K = 576`, `D = 1664`.

Cache và activation tăng gần tuyến tính theo `D`, nhưng compute trong predictor cũng tăng theo latent dim và predictor dim. Vì vậy chuyển lên model lớn hơn chỉ nên làm khi:

- Feature extraction bù đã hoàn tất.
- Train loop base ổn định.
- Data sạch.
- Có đủ VRAM.
- Có baseline tiny/small/base để so sánh.

### 3. Predictor official quá lớn cho thử nghiệm nhanh

Official dùng depth 24, dim 1024, heads 16. Với xe RC và GPU local 16GB, cấu hình này không thực dụng nếu giữ full-token 384. Vì vậy NN-JEPA dùng `tiny`, `small`, `base` là đúng về mặt workflow:

- `tiny`: kiểm tra pipeline, loss, W&B, resume, eval, inference.
- `small`: thử ablation nghiêm túc hơn.
- `base`: train chính khi mọi thứ đã sạch.

### 4. Không nên train pooled và full-token lẫn lộn

Hai nhánh này cần model/train script khác nhau:

- Full-token input: `[B, T*K, D]`.
- Pooled input: `[B, T, D]` hoặc `[B, T*1, D]`.

Nếu dùng pooled feature cho predictor full-token hiện tại thì sai về ý nghĩa. May mắn là DataLoader hiện tại yêu cầu `.npy` 3D `[N, K, D]`, nên pooled 2D `[N, D]` sẽ bị lỗi sớm.

## Đánh giá "đã đúng chuẩn V-JEPA 2.1-AC chưa?"

Câu trả lời chính xác:

> Chưa thể gọi là official V-JEPA 2.1-AC exact, vì repo public không cung cấp một bản V-JEPA 2.1-AC clean riêng để clone y hệt. Nhưng NN-JEPA hiện tại đang bám đúng nguyên lý quan trọng của V-JEPA-AC: frozen target encoder, token-level latent, action/state-conditioned predictor, teacher forcing + autoregressive rollout.

Nếu muốn tiến gần official hơn, thứ tự nâng cấp hợp lý là:

1. Giữ full-token ViT-B 384 và hoàn tất pipeline ổn định.
2. Train tiny/small/base để kiểm tra data, loss, eval, inference.
3. Thêm predictor kiểu official hơn:
   - action-block causal attention mask.
   - RoPE.
   - conditioning token layout giống `VisionTransformerPredictorAC`.
4. Sau khi predictor ổn, thử ViT-L 384.
5. Chỉ thử ViT-g/G khi có đủ VRAM/thời gian và có lý do rõ ràng.

Không nên nhảy thẳng lên ViT-G chỉ vì encoder mạnh hơn. Với xe RC indoor, lỗi thường đến từ data sync, action alignment, sensor noise, outlier, trajectory diversity và predictor design trước khi đến giới hạn encoder.

## Checklist hiện tại cần nhớ

Trước khi train nghiêm túc:

- Kiểm tra GPU hoạt động bằng `nvidia-smi`.
- Kiểm tra manifest và feature cache khớp đủ session.
- Không train nếu feature cache còn thiếu `.json` hoặc `.npy` cho session trong manifest.
- Dùng `eval_batch_size = 2` nếu full-token để tránh OOM.
- Dùng `model_size=tiny` hoặc `small` để smoke test trước.
- Chỉ dùng `base` khi smoke test đã pass.
- Không dùng pooled cache cho script full-token.
- Không sửa source gốc `vjepa2/` nếu chỉ đang train NN-JEPA.

## Kết luận cuối

Nếu mục tiêu là "giống paper/source official nhất có thể" thì hướng đúng của NN-JEPA hiện tại là giữ full-token latent, không mean-pool. Pooled latent là hướng thực dụng để train nhanh và debug nhanh, nhưng không phải chuẩn V-JEPA 2-AC official.

NN-JEPA hiện tại đã đúng ở phần xương sống quan trọng: frozen V-JEPA 2.1 encoder, token-level feature cache, action/state conditioning, teacher forcing loss, rollout loss. Phần chưa exact official nằm chủ yếu ở predictor architecture, attention mask, checkpoint AC public và schema robot/state/action. Đây là khác biệt có chủ đích để pipeline chạy được trên data xe RC và phần cứng local, nhưng cần ghi rõ để không nhầm là đã reproduce exact official V-JEPA 2-AC.
