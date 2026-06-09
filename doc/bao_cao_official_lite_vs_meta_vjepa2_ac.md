# Báo cáo: `official_lite` của NN-JEPA khác gì `VisionTransformerPredictorAC` của Meta V-JEPA2-AC?

Ngày viết: 2026-06-09

## Kết luận ngắn

`official_lite` hiện tại trong NN-JEPA **không giống y chang 100%** `VisionTransformerPredictorAC` của Meta.

Nó là bản **official-style / official-lite**:

- giữ lại ý tưởng cốt lõi của V-JEPA2-AC;
- giữ token layout `[action, state, patch tokens]`;
- giữ action-block causal attention mask;
- giữ RoPE cho token không gian/thời gian;
- output vẫn là latent patch tokens;
- train bằng `teacher_forcing_loss + rollout_loss`;
- nhưng bỏ bớt nhiều option/phần nặng của Meta để dễ đọc, dễ debug, dễ train trên data xe RC.

Nếu muốn nói thật chính xác:

```text
VJepaStyleACPredictor ~= bản rút gọn theo hướng VisionTransformerPredictorAC
VJepaStyleACPredictor != bản copy exact của Meta
```

## Source được đối chiếu

Source Meta V-JEPA2-AC:

```text
vjepa2/src/models/ac_predictor.py
vjepa2/src/models/utils/modules.py
vjepa2/configs/train/vitg16/droid-256px-8f.yaml
```

Source NN-JEPA:

```text
src/models/rc_jepa_ac.py
src/tools/train_rc_jepa_ac_features.py
src/data/feature_sequence_dataset.py
```

Class Meta:

```text
VisionTransformerPredictorAC
```

Class NN-JEPA:

```text
VJepaStyleACPredictor
```

## Mục tiêu của hai class

### Meta `VisionTransformerPredictorAC`

`VisionTransformerPredictorAC` là predictor action-conditioned chính thức trong public repo V-JEPA2 cho pipeline robot AC.

Nó nhận:

```text
x         = latent patch tokens
actions   = action tokens
states    = state tokens
extrinsics = optional camera extrinsics
```

Nó trả về:

```text
predicted latent patch tokens
```

Nó không trực tiếp trả về action.

### NN-JEPA `VJepaStyleACPredictor`

`VJepaStyleACPredictor` là bản predictor được viết lại trong NN-JEPA để chạy với feature cache của xe RC.

Nó nhận:

```text
latent_tokens = latent patch tokens từ V-JEPA 2.1 frozen encoder
actions       = action xe RC
states        = state xe RC
```

Nó trả về:

```text
predicted latent patch tokens
```

Nó cũng không trực tiếp trả về action.

## Những điểm giống nhau

### 1. Cùng là action-conditioned latent predictor

Cả hai đều là world-model predictor trên latent space.

Thay vì học:

```text
image -> action
```

chúng học:

```text
latent hiện tại + action/state -> latent tương lai
```

Điều này đúng tinh thần V-JEPA2-AC: predictor học dynamics trong latent space.

### 2. Cùng token layout chính

Cả hai đều chèn token điều kiện trước patch tokens của từng frame.

Không dùng extrinsics:

```text
[action token, state token, patch_1, patch_2, ..., patch_K]
```

Nếu lặp theo thời gian:

```text
frame 0: [a_0, s_0, patches_0]
frame 1: [a_1, s_1, patches_1]
frame 2: [a_2, s_2, patches_2]
...
```

Meta code:

```python
s = self.state_encoder(states).unsqueeze(2)
a = self.action_encoder(actions).unsqueeze(2)
x = x.view(B, T, self.grid_height * self.grid_width, D)
x = torch.cat([a, s, x], dim=2).flatten(1, 2)
```

NN-JEPA code:

```python
latent = self.predictor_embed(latent_tokens)
latent = latent.view(batch_size, num_frames, tokens_per_frame, self.predictor_dim)
action = self.action_encoder(actions).unsqueeze(2)
state = self.state_encoder(states).unsqueeze(2)
sequence = torch.cat([action, state, latent], dim=2).flatten(1, 2)
```

Về ý tưởng layout, hai bên giống nhau.

### 3. Cùng action-block causal attention mask

Cả hai đều dùng mask theo block thời gian.

Ý nghĩa:

- token ở frame hiện tại được nhìn frame hiện tại và quá khứ;
- không được nhìn tương lai;
- action/state/patch trong cùng frame nằm cùng block;
- đây là causal theo thời gian, không phải random mask kiểu pretrain JEPA.

Meta dùng:

```python
build_action_block_causal_attention_mask(
    grid_depth, grid_height, grid_width, add_tokens=3 if use_extrinsics else 2
)
```

NN-JEPA dùng:

```python
build_action_block_causal_attention_mask(
    num_frames=max_frames,
    grid_height=self.grid_height,
    grid_width=self.grid_width,
    add_tokens=self.cond_tokens,
)
```

Với case NN-JEPA hiện tại:

```text
raw_frames_per_sample = 8
tokens_per_frame = 576 = 24 x 24
cond_tokens = 2
tokens_per_step = 578
total sequence length = 8 x 578 = 4624
mask shape = [4624, 4624]
```

### 4. Cùng output contract

Cả hai sau transformer đều bỏ action/state tokens, chỉ lấy patch-token predictions.

Meta:

```python
x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)
x = x[:, :, cond_tokens:, :].flatten(1, 2)
x = self.predictor_norm(x)
x = self.predictor_proj(x)
return x
```

NN-JEPA:

```python
sequence = sequence.view(batch_size, num_frames, self.cond_tokens + tokens_per_frame, self.predictor_dim)
predicted = sequence[:, :, self.cond_tokens:, :].flatten(1, 2)
predicted = self.predictor_proj(self.predictor_norm(predicted))
return predicted
```

Về output shape/ý nghĩa, hai bên tương đương:

```text
[B, T * tokens_per_frame, latent_dim]
```

### 5. Cùng kiểu khởi tạo chính

Cả hai dùng:

- truncated normal cho `Linear.weight`;
- bias bằng 0;
- LayerNorm weight bằng 1, bias bằng 0;
- rescale attention projection và MLP cuối theo layer id.

Meta:

```python
trunc_normal_(m.weight, std=self.init_std)
rescale(layer.attn.proj.weight.data, layer_id + 1)
rescale(layer.mlp.fc2.weight.data, layer_id + 1)
```

NN-JEPA:

```python
nn.init.trunc_normal_(module.weight, std=0.02)
block.attn.proj.weight.data.div_(math.sqrt(2.0 * layer_id))
block.mlp.fc2.weight.data.div_(math.sqrt(2.0 * layer_id))
```

## Những điểm khác nhau quan trọng

## 1. Đây không phải cùng class

Meta:

```text
VisionTransformerPredictorAC
```

NN-JEPA:

```text
VJepaStyleACPredictor
```

NN-JEPA không import trực tiếp class official từ `vjepa2/src/models/ac_predictor.py`.

Thay vào đó, NN-JEPA tự viết lại các phần cần thiết:

```text
VJepaStyleACPredictor
VJepaStyleACBlock
VJepaStyleACAttention
VJepaStyleMLP
build_action_block_causal_attention_mask
rotate_queries_or_keys
```

Lý do viết lại:

- dễ đọc hơn;
- không phụ thuộc sâu vào internal package layout của `vjepa2`;
- dễ chỉnh `state_dim/action_dim` cho xe RC;
- tránh kéo theo nhiều option nặng;
- dễ test shape/mask/RoPE trong repo NN-JEPA.

Tác dụng phụ:

- không đảm bảo giống từng dòng với Meta;
- nếu Meta update implementation, NN-JEPA không tự update theo;
- có rủi ro sai khác nhỏ ở attention/DropPath/MLP nếu muốn reproduction exact.

## 2. Khác kích thước predictor

Meta default trong code:

```text
predictor_embed_dim = 1024
depth = 24
num_heads = 16
```

Meta public DROID config:

```text
pred_depth = 24
pred_embed_dim = 1024
pred_num_heads = 16
```

NN-JEPA có preset nhẹ:

```text
tiny:  predictor_dim = 128, depth = 2, heads = 4
small: predictor_dim = 256, depth = 4, heads = 4
base:  predictor_dim = 512, depth = 6, heads = 8
```

Với feature ViT-B 384 hiện tại:

```text
official_lite tiny  ~= 0.595M params
official_lite small ~= 3.556M params
official_lite base  ~= 19.708M params
```

Meta predictor official trong config robot lớn hơn rất nhiều.

Tác động:

- NN-JEPA nhẹ hơn, train nhanh hơn, ít VRAM hơn.
- Nhưng capacity thấp hơn Meta official.
- Nếu data phức tạp hơn, `official_lite` có thể underfit so với official lớn.

## 3. Khác action/state dimension

Meta `VisionTransformerPredictorAC` dùng:

```python
action_embed_dim=7
self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim)
```

Tức là state và action trong pipeline robot public cùng dùng dimension 7.

NN-JEPA dùng:

```python
action_dim = len(action_columns)
state_dim = len(state_columns)
self.action_encoder = nn.Linear(action_dim, predictor_dim)
self.state_encoder = nn.Linear(state_dim, predictor_dim)
```

Hiện tại:

```text
state_dim = 5
state_columns = [yaw_rate_t, accel_x_t, accel_y_t, steering_last_t, throttle_last_t]

action_dim = 2
action_columns = [steering_cmd_t, throttle_cmd_t]
```

Tác động:

- NN-JEPA phù hợp xe RC hơn.
- Nhưng không giống exact DROID robot setup.
- Không thể load weight predictor official của Meta trực tiếp vào NN-JEPA vì shape encoder action/state khác.

## 4. Meta có extrinsics, NN-JEPA hiện chưa có

Meta có option:

```python
use_extrinsics=False
self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim)
```

Nếu bật `use_extrinsics=True`, layout trở thành:

```text
[action token, state token, extrinsics token, patch tokens]
```

NN-JEPA hiện không có extrinsics token.

Trong xe RC hiện tại, điều này hợp lý vì:

- chỉ có một camera cố định;
- chưa có calibration camera extrinsics chuẩn;
- chưa có nhiều camera/góc nhìn như DROID;
- extrinsics giả hoặc sai có thể gây nhiễu.

Extrinsics nghĩa là thông tin hình học bên ngoài của camera:

```text
camera nằm ở đâu
camera xoay hướng nào
camera liên hệ với robot/world frame ra sao
```

Trong robotics, extrinsics thường là pose/transform giữa camera frame và robot/world frame.

## 5. Meta có DropPath, NN-JEPA hiện chưa có

Meta block có `drop_path_rate`:

```python
dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
Block(... drop_path=dpr[i] ...)
```

Trong `vjepa2/src/models/utils/modules.py`, `DropPath` là stochastic depth.

NN-JEPA hiện không có DropPath trong `VJepaStyleACBlock`.

NN-JEPA chỉ có:

```text
dropout trong attention projection
dropout trong MLP
```

Không có:

```text
drop_path_rate
stochastic depth
```

Tác động:

- Với `tiny` 2 layer: gần như không quan trọng.
- Với `small` 4 layer: thường chưa quan trọng.
- Với `base` 6 layer: có thể hữu ích nếu overfit.
- Với predictor sâu 12-24 layer: DropPath quan trọng hơn.

Khuyến nghị hiện tại:

```text
tiny/small: drop_path_rate = 0.0
base:       có thể thử 0.05 nếu train loss thấp nhưng val loss kém
deep 12+:   nên cân nhắc 0.1 trở lên
```

Chưa nên thêm DropPath nếu:

```text
train loss còn cao
val loss cũng cao
pipeline data/model còn đang debug
đang ưu tiên kiểm tra inference/planner
```

## 6. Meta có activation checkpointing, NN-JEPA chưa có

Meta có:

```python
use_activation_checkpointing
torch.utils.checkpoint.checkpoint(...)
```

Mục đích:

- giảm VRAM khi train predictor sâu;
- đổi lại train chậm hơn vì phải recompute forward trong backward.

NN-JEPA hiện chưa có activation checkpointing trong `official_lite`.

Tác động:

- Code đơn giản hơn.
- Nhưng nếu tăng predictor lên sâu hơn, VRAM sẽ căng hơn.
- Với `base` 6 layer, chưa bắt buộc.
- Nếu muốn chạy official-depth 24 layer, activation checkpointing gần như nên có.

## 7. Meta có nhiều option MLP/activation hơn

Meta block support:

```text
GELU
SiLU
wide_silu
SwiGLU-like FFN option trong modules
drop rate
attention drop rate
qkv_bias
qk_scale
drop_path
```

NN-JEPA hiện dùng MLP đơn giản:

```python
Linear -> GELU -> Dropout -> Linear -> Dropout
```

Tác động:

- NN-JEPA dễ đọc hơn.
- Ít option hơn để tune.
- Nhưng không giống exact Meta block.

## 8. Attention implementation được viết lại

Meta dùng:

```text
ACBlock
ACRoPEAttention
```

trong:

```text
vjepa2/src/models/utils/modules.py
```

NN-JEPA dùng:

```text
VJepaStyleACBlock
VJepaStyleACAttention
```

Trong NN-JEPA, attention logic đã được viết lại theo source Meta:

- tách action/state tokens;
- tính QKV riêng;
- apply RoPE cho action tokens theo frame position;
- apply RoPE cho patch tokens theo frame/height/width;
- gộp action heads và patch heads;
- dùng `scaled_dot_product_attention`.

Nhưng vì là code tự viết lại, nó không phải exact same class.

Tác động:

- Nếu test shape/mask/RoPE pass thì đủ tốt cho hướng official-lite.
- Nếu mục tiêu là reproduction exact, nên import trực tiếp class Meta thay vì tự viết lại.

## 9. Khác cách hiểu `num_frames`, `tubelet_size`, và feature cache

Meta code build mask bằng:

```python
grid_depth = self.num_frames // self.tubelet_size
```

Trong pipeline DROID/video, `num_frames` và `tubelet_size` liên hệ trực tiếp với video tubelets.

NN-JEPA hiện train từ feature cache theo từng frame thật.

Feature extractor hiện làm:

```text
mỗi frame thật -> pseudo clip có tubelet_size=2 frame duplicate
encoder trả ra token cho frame đó
cache lưu [N, 576, 768]
```

Khi train predictor từ cache:

```text
1 sample = 8 frame thật
tokens_per_frame = 576
total latent tokens = 8 x 576 = 4608
```

NN-JEPA `official_lite` build mask theo:

```text
num_frames = raw_frames_per_sample = 8
grid = 24 x 24
```

Điều này hợp với feature cache hiện tại, nhưng không giống hoàn toàn cách official video/tubelet path được tổ chức.

## 10. Khác encoder/feature setup

Meta public robot AC config:

```text
model_name = vit_giant_xformers
crop_size = 256
tubelet_size = 2
dataset_fpcs = 8
pred_depth = 24
pred_embed_dim = 1024
pred_num_heads = 16
dtype = bfloat16
```

NN-JEPA hiện tại:

```text
encoder = V-JEPA 2.1 ViT-B 384
image_size = 384
patch_size = 16
tokens_per_frame = 576
embed_dim = 768
dtype feature cache = fp32
predictor = official_lite tiny/small/base
```

Tác động:

- NN-JEPA nhẹ hơn và hợp GPU nhỏ hơn.
- Nhưng không phải public DROID config.
- Không thể so sánh trực tiếp với Meta official nếu encoder/predictor/data khác.

## 11. Khác training integration

Meta public train loop robot AC dùng online target encoder trong `vjepa2/app/vjepa_droid/train.py`.

Ý tưởng chính:

```text
target_encoder(c) -> latent target
predictor(latent, action, state, optional extrinsics)
teacher forcing loss
autoregressive rollout loss
```

NN-JEPA feature-cache train:

```text
extract feature trước bằng frozen V-JEPA 2.1 encoder
lưu .npy
train predictor từ latents đã cache
```

Tác động:

- NN-JEPA train nhanh hơn vì không chạy encoder trong train loop.
- Nhưng không còn augmentation ảnh online.
- Nếu đổi encoder/checkpoint/image size thì phải extract lại feature và train predictor lại.

## 12. Khác handling state rollout

Meta robot dùng state/action/extrinsics theo robot trajectory.

NN-JEPA rollout hiện dùng:

```text
initial state lặp lại cho future steps
copy previous action vào steering_last_t/throttle_last_t
```

Tức là:

- không dự đoán IMU tương lai bằng physics model;
- không có full vehicle dynamics;
- giữ state rollout đơn giản để khớp train loop.

Điều này là tradeoff thực dụng.

Nếu muốn mạnh hơn kiểu JEPA repo mới:

- thêm full IMU state;
- fit dynamics đơn giản;
- rollout future state bằng `CarDynamics`;
- hoặc train một state transition model riêng.

## Bảng so sánh nhanh

| Hạng mục | Meta `VisionTransformerPredictorAC` | NN-JEPA `VJepaStyleACPredictor` |
|---|---|---|
| Mục tiêu | Action-conditioned latent predictor | Action-conditioned latent predictor |
| Output | Patch latent tokens | Patch latent tokens |
| Layout | `[action,state,(extrinsics),patches]` | `[action,state,patches]` |
| Extrinsics | Có option | Chưa có |
| Action dim | `action_embed_dim`, default 7 | `action_dim=2` |
| State dim | dùng `action_embed_dim`, default 7 | `state_dim=5` |
| Mask | action-block causal | action-block causal |
| RoPE | Có | Có, viết lại |
| Block | Meta `ACBlock` | custom `VJepaStyleACBlock` |
| MLP | nhiều option, GELU/SiLU/SwiGLU-like | GELU MLP đơn giản |
| DropPath | Có | Chưa có |
| Activation checkpointing | Có | Chưa có |
| Config public robot | depth 24, dim 1024, heads 16 | tiny/small/base, max base depth 6 dim 512 |
| Encoder public robot | thường ViT-g 256 | hiện ViT-B 384 |
| Training | online encoder trong train loop | feature cache frozen encoder |
| Mục tiêu repo | robot DROID / public AC | xe RC indoor |

## DropPath có nên thêm không?

Chưa cần thêm ngay.

DropPath hữu ích khi predictor sâu và dễ overfit. Với model hiện tại:

```text
tiny  = 2 layer -> không cần
small = 4 layer -> thường chưa cần
base  = 6 layer -> có thể thử nếu overfit
```

Nên thêm DropPath khi thấy:

```text
train/loss giảm thấp
val/loss đứng hoặc tăng
train-val gap lớn
test/planner kém dù train tốt
```

Không nên thêm khi:

```text
train/loss còn cao
val/loss cũng cao
pipeline còn đang thay đổi
đang debug feature/inference/planner
```

Nếu thêm thì nên làm optional:

```text
drop_path_rate default = 0.0
```

và không resume lẫn checkpoint cũ khi bật `drop_path_rate > 0`.

## Extrinsics có nên thêm không?

Chưa nên thêm nếu chưa có camera calibration thật.

Extrinsics chỉ hữu ích nếu biết chính xác camera pose:

```text
vị trí camera so với xe
hướng camera
transform camera frame -> vehicle/world frame
```

Trong xe RC hiện tại:

- camera gần như cố định;
- chưa có calibration chuẩn;
- không có multi-camera setup;
- state/action hiện chưa cần extrinsics.

Nếu thêm extrinsics giả, model có thể học nhiễu.

Nên chỉ thêm khi:

- có nhiều camera;
- camera pose thay đổi;
- hoặc muốn làm navigation có tọa độ/geometry rõ hơn.

## Nếu muốn giống Meta hơn thì nên làm gì?

Thứ tự hợp lý:

### Bước 1: Giữ `official_lite`, train/eval cho ổn

Trước mắt nên dùng:

```text
official_lite tiny
official_lite small
```

để kiểm tra:

- loss train/val;
- OOM;
- checkpoint/resume;
- offline planner CEM;
- output đồ thị planner.

### Bước 2: Thêm DropPath optional

Nếu `base` overfit, thêm:

```text
drop_path_rate
```

vào `VJepaStyleACBlock`.

Default vẫn phải là:

```text
0.0
```

để không phá checkpoint cũ.

### Bước 3: Thêm activation checkpointing nếu tăng depth

Nếu muốn thử:

```text
depth >= 12
```

nên thêm activation checkpointing để giảm VRAM.

### Bước 4: Cân nhắc import trực tiếp `VisionTransformerPredictorAC`

Nếu mục tiêu là official-exact hơn, có thể thêm predictor type mới:

```text
predictor_type = meta_ac
```

Ý tưởng:

```python
from vjepa2.src.models.ac_predictor import VisionTransformerPredictorAC
```

Nhưng cần xử lý cẩn thận:

- Python import path của `vjepa2`;
- state/action dim;
- `img_size`;
- `num_frames`;
- `tubelet_size`;
- `use_extrinsics`;
- checkpoint compatibility;
- memory/OOM.

Không nên thay thế `official_lite` trực tiếp. Nên thêm như option mới.

### Bước 5: Port state/dynamics tốt hơn

Để inference/planner mạnh hơn, chỉ đổi predictor chưa đủ.

Cần cải thiện:

- state vector;
- action scaling;
- future state rollout;
- dynamics hoặc state transition;
- goal selection/topological graph.

Repo `JEPA/` mới có hướng `VJEPA2ACCar` dùng full IMU 10D và CEM planner với dynamics. Đây là hướng đáng học hỏi, nhưng không nên trộn checkpoint/feature trực tiếp vì format khác NN-JEPA hiện tại.

## Kết luận cuối

`VJepaStyleACPredictor` hiện tại là lựa chọn đúng cho giai đoạn này vì:

- đủ gần official ở cấu trúc quan trọng;
- nhẹ hơn nhiều;
- dễ đọc;
- dễ debug;
- hợp feature cache NN-JEPA hiện tại;
- không phụ thuộc quá sâu vào source Meta;
- dễ chạy thử nhiều experiment.

Nhưng cần ghi rõ:

```text
official_lite không phải official-exact
```

Nếu mục tiêu nghiên cứu là bám Meta nhất có thể, cần thêm một nhánh `meta_ac` hoặc `official_exact` riêng, không thay thế bản hiện tại.

Nếu mục tiêu là làm xe RC chạy được ổn định, ưu tiên hiện tại nên là:

```text
data đúng -> feature đúng -> official_lite/tiny chạy ổn -> planner offline có đồ thị -> live dry-run -> closed-loop an toàn
```

chứ chưa nên nhảy ngay sang predictor 24 layer giống Meta.
