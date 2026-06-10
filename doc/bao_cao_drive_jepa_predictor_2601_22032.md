# Báo Cáo Predictor Của Drive-JEPA Theo Paper 2601.22032v1 Và Source Code

Ngày rà soát: 2026-06-10

Repo được đọc:

```text
Drive-JEPA/
```

Paper được đọc:

```text
doc/2601.22032v1.pdf
```

Các file code chính đã đối chiếu:

```text
Drive-JEPA/README.md
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_agent.py
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_config.py
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py
Drive-JEPA/navsim_v2/vjepa2/configs/train/vitg16/droid-256px-8f.yaml
```

## 1. Kết Luận Ngắn Gọn

Drive-JEPA có hai thứ rất dễ bị gọi chung là `predictor`, nhưng chúng khác vai trò.

Predictor thứ nhất là `V-JEPA predictor`.

Đây là predictor đúng nghĩa trong kiến trúc JEPA. Nó học dự đoán latent representation của video ở vùng bị mask hoặc vùng tương lai. Nó hoạt động trong giai đoạn self-supervised video pretraining. Code tương ứng nằm ở:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py
```

Predictor thứ hai là `trajectory planner` hoặc `trajectory decoder`.

Đây là phần dùng khi chạy bài toán lái xe end-to-end. Nó nhận feature từ encoder V-JEPA hoặc ViT, nhận ego status, rồi sinh trajectory hoặc nhiều trajectory proposals. Phần này mới là head trực tiếp tạo đường đi cho xe.

Code perception-free nằm ở:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py
```

Code perception-based nằm ở:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py
```

Kết luận quan trọng:

```text
Drive-JEPA không dùng V-JEPA AC predictor để trực tiếp xuất steering/throttle hay trajectory.
Drive-JEPA dùng V-JEPA để pretrain hoặc lấy representation ảnh/video.
Sau đó nó gắn một trajectory decoder/planner riêng để sinh waypoint trajectory.
```

Vì vậy nếu so với NN-JEPA hiện tại:

```text
NN-JEPA hiện tại đang đi theo hướng world-model latent predictor:
  latent_t + state/action -> latent_future

Drive-JEPA đi theo hướng planning head:
  image/video feature + ego status -> trajectory/proposals -> chọn trajectory
```

Hai hướng này cùng dùng JEPA representation, nhưng mục tiêu train khác nhau.

## 2. Repo Drive-JEPA Có Những Phần Nào

Repo có hai nhánh chính:

```text
Drive-JEPA/navsim_v1/
Drive-JEPA/navsim_v2/
```

Mỗi nhánh có:

```text
navsim/
vjepa2/
scripts/
```

Ý nghĩa:

```text
navsim/
```

Là phần agent, planner, dataloader, metric, training theo NAVSIM.

```text
vjepa2/
```

Là fork hoặc copy từ V-JEPA2. Trong đó có encoder, predictor, train config, eval config.

```text
scripts/
```

Là các script train, cache feature, eval.

Điểm dễ nhầm:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py
```

File này có `VisionTransformerPredictorAC`, nhưng đó là predictor kiểu V-JEPA2-AC, không phải toàn bộ Drive-JEPA planner.

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py
```

File này mới là model planner chính trong setting perception-based.

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py
```

File này là model planner đơn giản trong setting perception-free.

## 3. Paper Nói Gì Về Predictor

Trong paper `doc/2601.22032v1.pdf`, Drive-JEPA được mô tả gồm ba module:

```text
1. Driving Video Pretraining
2. Multimodal Trajectory Distillation
3. Momentum-aware Trajectory Selection
```

Paper nói V-JEPA dùng kiến trúc:

```text
encoder Eθ
predictor Pϕ
EMA target encoder Eθ̄
stop-gradient target
loss trên latent target bị mask
```

Mục tiêu V-JEPA:

```text
Pϕ(mask token, Eθ(context video)) ≈ sg(Eθ̄(target video))
```

Ý nghĩa:

```text
Eθ encode phần context của video.
Eθ̄ encode target video bằng EMA encoder.
Pϕ dự đoán latent target từ context latent.
Loss được tính trong không gian latent, không reconstruct pixel.
```

Đây là self-supervised pretraining. Nó giúp encoder học representation tốt cho driving.

Paper sau đó dùng encoder đã pretrain để làm planning:

```text
front-view image/video
-> ViT encoder
-> spatiotemporal features
-> transformer decoder hoặc proposal planner
-> future waypoints
```

Điểm chính:

```text
Predictor trong V-JEPA pretraining không phải là planner cuối.
Planner cuối là một module khác, dùng feature từ encoder.
```

## 4. V-JEPA Predictor Trong Source Code

File:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py
```

Class chính:

```python
class VisionTransformerPredictorAC(nn.Module):
    """Action Conditioned Vision Transformer Predictor"""
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:17
```

Đây là predictor action-conditioned của V-JEPA2-AC.

Input của forward:

```python
def forward(self, x, actions, states, extrinsics=None):
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:136
```

Ý nghĩa input:

```text
x
```

Latent context tokens từ vision encoder. Shape logic là:

```text
[B, N_ctxt, embed_dim]
```

Trong đó:

```text
B là batch size.
N_ctxt là số token context.
embed_dim là chiều latent của encoder.
```

```text
actions
```

Action token theo thời gian. Trong code default `action_embed_dim=7`.

```text
states
```

State token theo thời gian. Code dùng cùng `action_embed_dim=7`.

```text
extrinsics
```

Optional. Nếu bật `use_extrinsics=True`, nó thêm token extrinsics vào sequence.

## 5. Cấu Trúc VisionTransformerPredictorAC

Các thành phần chính:

```python
self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
self.action_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
self.state_encoder = nn.Linear(action_embed_dim, predictor_embed_dim, bias=True)
self.extrinsics_encoder = nn.Linear(action_embed_dim - 1, predictor_embed_dim, bias=True)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:53
```

Ý nghĩa:

```text
predictor_embed
```

Project latent encoder từ `embed_dim` sang `predictor_embed_dim`.

```text
action_encoder
```

Encode vector action thành token cùng chiều với predictor.

```text
state_encoder
```

Encode vector state thành token cùng chiều với predictor.

```text
extrinsics_encoder
```

Encode camera/robot extrinsics nếu dùng.

Transformer blocks:

```python
self.predictor_blocks = nn.ModuleList([... ACBlock ...])
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:79
```

Mỗi block là:

```python
ACBlock
```

Được import từ:

```python
from vjepa2.src.models.utils.modules import ACBlock as Block
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:12
```

Attention mask:

```python
build_action_block_causal_attention_mask(...)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:114
```

Ý nghĩa:

```text
Mask này làm attention theo kiểu causal theo block thời gian.
Mỗi frame có thêm action/state token.
Token ở tương lai không được nhìn lung tung về target không hợp lệ.
```

Forward flow:

```python
x = self.predictor_embed(x)
```

Project latent vào predictor dim.

```python
s = self.state_encoder(states).unsqueeze(2)
a = self.action_encoder(actions).unsqueeze(2)
```

Tạo state/action tokens.

```python
x = x.view(B, T, self.grid_height * self.grid_width, D)
```

Chia token thành frame dimension:

```text
[B, T, H*W, D]
```

Nếu không dùng extrinsics:

```python
x = torch.cat([a, s, x], dim=2).flatten(1, 2)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:153
```

Shape sau concat:

```text
[B, T*(H*W+2), D]
```

Nếu dùng extrinsics:

```text
[B, T*(H*W+3), D]
```

Sau Transformer:

```python
x = x.view(B, T, cond_tokens + self.grid_height * self.grid_width, D)
x = x[:, :, cond_tokens:, :].flatten(1, 2)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:183
```

Nó bỏ action/state/extrinsics tokens, chỉ giữ lại image latent tokens.

Cuối cùng:

```python
x = self.predictor_norm(x)
x = self.predictor_proj(x)
return x
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:187
```

Output:

```text
predicted latent tokens
```

Không phải steering, throttle, hay trajectory.

## 6. Thông Số Mặc Định Của VisionTransformerPredictorAC

Trong constructor:

```text
img_size=(224, 224)
patch_size=16
num_frames=1
tubelet_size=2
embed_dim=768
predictor_embed_dim=1024
depth=24
num_heads=16
mlp_ratio=4.0
drop_rate=0.0
attn_drop_rate=0.0
drop_path_rate=0.0
use_rope=True
action_embed_dim=7
use_extrinsics=False
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py:20
```

Khi build từ hub:

```text
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py
```

Function:

```python
def _make_vjepa2_ac_model(...)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py:27
```

Default AC model:

```text
model_name = "vit_ac_giant"
img_size = 256
patch_size = 16
tubelet_size = 2
num_frames = 64
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py:29
```

Checkpoint public:

```text
vjepa2-ac-vitg.pt
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py:17
```

Load encoder và predictor:

```python
encoder.load_state_dict(...)
predictor.load_state_dict(...)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/vjepa2/src/hub/backbones.py:75
```

## 7. Config V-JEPA2-AC Droid Trong Repo

File:

```text
Drive-JEPA/navsim_v2/vjepa2/configs/train/vitg16/droid-256px-8f.yaml
```

Các thông số đáng chú ý:

```yaml
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
  lr: 0.000425
  start_lr: 0.000075
  warmup: 15
```

Ý nghĩa với NN-JEPA:

```text
Official V-JEPA2-AC dùng token-level latent.
Nó không mean-pool token thành một vector nhỏ.
Nó dùng predictor khá lớn: depth 24, dim 1024, heads 16.
Nó dùng BF16 và activation checkpointing để giảm VRAM.
Nó dùng frame causal attention mask.
```

## 8. Perception-Free Drive-JEPA Predictor

Đây là bản đơn giản nhất trong Drive-JEPA để dùng V-JEPA representation cho planning.

File:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py
```

Class:

```python
class DriveJEPAModel(nn.Module):
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:21
```

Input:

```text
camera_feature
status_feature
```

Trong feature builder:

```text
camera_feature hoặc camera_feature_1/camera_feature_2
status_feature = driving_command + ego_velocity + ego_acceleration
```

Status feature shape:

```text
4 + 2 + 2 = 8
```

Vì code có:

```python
self._status_encoding = nn.Linear(4 + 2 + 2, tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:60
```

## 9. Cấu Trúc Perception-Free Model

Các thông số mặc định:

```text
image_architecture = "vit_large"
tf_d_model = 256
tf_d_ffn = 1024
tf_num_layers = 3
tf_num_head = 8
tf_dropout = 0.0
front_only = True
freeze_encoder = True
double_image = False
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:26
```

Load V-JEPA encoder:

```python
self.image_encoder = init_module(...)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:49
```

Nếu freeze encoder:

```python
self.image_encoder.eval()
for p in self.image_encoder.parameters():
    p.requires_grad = False
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:53
```

Project image feature:

```python
self.image_fc = nn.Linear(MODEL_DICT[image_architecture]["dim"], tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:59
```

Key/value embedding:

```python
self._keyval_embedding = nn.Embedding(num_keyval, tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:65
```

Waypoint query embedding:

```python
self._query_embedding = nn.Embedding(num_poses, tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:66
```

Transformer:

```python
self._transformer = nn.Transformer(...)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:68
```

Trajectory head:

```python
self._trajectory_head = TrajectoryHead(num_poses, tf_d_ffn, tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:77
```

Forward:

```python
img_feat = self.image_encoder(camera_feature)
img_feat = self.avg_pool(img_feat)
img_feat = self.image_fc(img_feat.clone())
status_encoding = self._status_encoding(status_feature)
keyval = torch.cat([img_feat, status_encoding[:, None]], dim=1)
query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
query_out = self._transformer(src=keyval_final, tgt=query)
trajectory = self._trajectory_head(query_out)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:96
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:108
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:114
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:115
```

Output:

```text
trajectory [B, num_poses, 3]
```

Trong đó mỗi waypoint:

```text
x, y, heading
```

Heading được clamp bằng:

```python
poses[..., StateSE2Index.HEADING] = poses[..., StateSE2Index.HEADING].tanh() * np.pi
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py:145
```

## 10. Loss Của Perception-Free Model

File:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_agent.py
```

Loss:

```python
return l1_length_normalized_loss(pred, gt, alpha=5.0)
```

Vị trí:

```text
Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_agent.py
```

Ý nghĩa:

```text
pred là trajectory model dự đoán.
gt là human future trajectory.
Loss là L1 nhưng được normalize theo độ dài quỹ đạo ground-truth.
```

Mục tiêu:

```text
Học trực tiếp từ human trajectory.
Không học latent dynamics.
Không dự đoán feature tương lai.
Không có action-conditioned world model ở phần planner này.
```

## 11. Perception-Based Drive-JEPA Planner

Đây là bản đầy đủ hơn, đúng với phần “proposal-centric planner” trong paper.

File:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py
```

Class:

```python
class DriveJEPAModel(nn.Module):
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:24
```

Input chính:

```text
camera_feature_1
camera_feature_2
ego_status
lidar2img
metric_cache optional khi calibrate score
past_ego_simulated_states optional khi momentum-aware score
```

Pipeline:

```text
2 ảnh front camera
-> normalize ImageNet
-> V-JEPA/ViT image backbone
-> ego status Linear
-> tạo learnable proposal feature
-> refine proposal nhiều lần
-> scorer chấm điểm proposal
-> chọn proposal score cao nhất
-> trajectory cuối
```

## 12. Cấu Trúc Perception-Based Model

Backbone:

```python
self._backbone = ImgEncoder(config)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:31
```

Ego status encoder:

```python
self.hist_encoding = nn.Linear(11, config.tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:32
```

Proposal feature init:

```python
self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:33
```

Nếu:

```text
poses_num = 8
proposal_num = 32
tf_d_model = 256
```

Thì initial proposal feature có:

```text
8 * 32 = 256 tokens
```

Mỗi token có chiều:

```text
256
```

Refiner:

```python
shared_refiner = Traj_refiner(config)
self._trajectory_head = nn.ModuleList([shared_refiner for _ in range(config.ref_num)])
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:35
```

Scorer:

```python
self.scorer = Scorer(config)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:38
```

Forward:

```python
image_feature = self._backbone(camera_feature, img_metas=features)
ego_feature = self.hist_encoding(ego_status)[:, None]
bev_feature = ego_feature + self.init_feature.weight[None]
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:60
```

Lặp refinement:

```python
for _, refine in enumerate(self._trajectory_head):
    bev_feature, proposals = refine(bev_feature, image_feature)
    proposal_list.append(proposals)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:65
```

Scoring:

```python
pred_logit, ... = self.scorer(proposals, bev_feature)
pdm_score = torch.sigmoid(pred_logit)[:, :, -1]
token = torch.argmax(pdm_score, dim=1)
trajectory = proposals[torch.arange(batch_size), token]
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:71
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:85
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:90
```

Output:

```text
proposals
proposal_list
pred_logit
pred_agents_states
pred_area_logit
trajectory
pdm_score
```

## 13. Traj_refiner Là Gì

File:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py
```

Class:

```python
class Traj_refiner(nn.Module):
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py:8
```

Thành phần:

```python
self.traj_decoder = MLP(config.tf_d_model, config.tf_d_ffn, self.state_size)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py:21
```

Forward:

```python
proposals = self.traj_decoder(bev_feature).reshape(
    bev_feature.shape[0],
    -1,
    self.poses_num,
    self.state_size,
)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py:24
```

Shape:

```text
[B, proposal_num, num_poses, 3]
```

Với default:

```text
[B, 32, 8, 3]
```

Sau đó nếu `traj_bev=True`:

```python
bev_feature = self.Bev_refiner(proposals, bev_feature, image_feature)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py:27
```

Ý nghĩa:

```text
Nó sinh proposal từ feature hiện tại.
Sau đó dùng proposal làm anchor để query/refine feature ảnh qua BEV/deformable attention.
Lặp nhiều lần để proposal ngày càng tốt hơn.
```

## 14. Scorer Là Gì

File:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py
```

Class:

```python
class Scorer(nn.Module):
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:8
```

Scorer head:

```python
self.pred_score = MLP(config.tf_d_model, config.tf_d_ffn, self.score_num)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:15
```

Default:

```text
score_num = 6
```

Forward:

```python
proposal_feature = bev_feature.reshape(batch_size, p_size, t_size, -1).amax(-2)
pred_logit = self.pred_score(proposal_feature).reshape(batch_size, -1, self.score_num)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:41
```

Ý nghĩa:

```text
bev_feature có feature theo từng proposal và từng waypoint.
Nó max-pool qua waypoint dimension.
Sau đó MLP dự đoán score/logits cho từng proposal.
```

Khi train, nó còn có auxiliary heads:

```text
pred_col_agent
pred_area
bev_map optional
bev_agent optional
```

Trong code:

```text
pred_col_agent dùng để dự đoán collision-related agent states.
pred_area dùng để dự đoán route/area logits.
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:19
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:24
```

## 15. Loss Của Perception-Based Model

File:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py
```

Loss tổng trong `pad_loss`:

```python
loss = (
    config.trajectory_weight * trajectory_loss
    + config.sub_score_weight * sub_score_loss
    + config.final_score_weight * final_score_loss
    + config.pred_ce_weight * pred_ce_loss
    + config.pred_l1_weight * pred_l1_loss
    + config.pred_area_weight * pred_area_loss
    + config.agent_class_weight * agent_class_loss
    + config.agent_box_weight * agent_box_loss
    + config.bev_semantic_weight * bev_semantic_loss
)
```

Ý nghĩa:

```text
trajectory_loss
```

Học proposal gần human trajectory và pseudo-teacher trajectory.

```text
sub_score_loss
```

Học score phụ từ simulator metrics.

```text
final_score_loss
```

Học final score của proposal.

```text
pred_ce_loss, pred_l1_loss
```

Auxiliary collision/agent prediction.

```text
pred_area_loss
```

Auxiliary area/route prediction.

```text
agent_class_loss, agent_box_loss, bev_semantic_loss
```

Optional auxiliary perception losses.

Điểm quan trọng trong code:

```python
proposals = proposals.detach()
trajectory_loss, ... = self.trajectory_loss_anchors(...)
```

Trong `pad_loss`, sau khi compute score, final proposals được detach ở một đoạn. Nhưng `proposal_list` vẫn được dùng để train trajectory refiners.

Paper mô tả loss:

```text
L = Ltraj + wscore Lscore + wmap Lmap + wcolli Lcolli
```

Source code triển khai cùng tinh thần, nhưng tên biến cụ thể khác:

```text
trajectory_loss tương ứng Ltraj.
final_score_loss và sub_score_loss tương ứng score supervision.
pred_area_loss tương ứng area/map style auxiliary.
pred_ce_loss/pred_l1_loss tương ứng collision/agent auxiliary.
```

## 16. Multimodal Trajectory Distillation Trong Code

Paper nói:

```text
Tạo trajectory vocabulary bằng clustering.
Chọn 8192 centers.
Dùng simulator chấm EPDM/PDM score.
Lấy trajectory tốt làm pseudo-teacher.
Train proposal distribution không collapse vào một human trajectory duy nhất.
```

Trong code:

```python
poses = np.load("./data/8192.npy")
self.anchors = poses[:, 4::5]
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py
```

Trong `trajectory_loss_anchors`, code dùng `scores_index` để chọn anchor pseudo targets:

```python
anchors_np = self.anchors[sampled_idx]
pseudo_targets = torch.from_numpy(anchors_np).to(...)
```

Ý nghĩa:

```text
anchors là vocabulary trajectory.
scores_index là các trajectory anchor được simulator đánh giá tốt cho scene hiện tại.
Pseudo targets được dùng thêm vào min-over-proposal loss.
```

Điểm quan trọng:

```text
Nó không chỉ imitation theo human trajectory.
Nó cố làm proposal distribution đa dạng hơn bằng các pseudo-teacher trajectories.
```

Đây là phần khác biệt lớn giữa Drive-JEPA và imitation learning thường.

## 17. Momentum-Aware Trajectory Selection

Paper nói:

```text
MTD làm proposal đa dạng hơn, nhưng có thể làm trajectory giữa các frame bị nhảy.
Momentum-aware selection thêm comfort term để giảm thay đổi đột ngột giữa frame t-1 và t.
```

Trong code perception-based:

```python
if 'past_ego_simulated_states' in features and features['past_ego_simulated_states'] is not None:
    pdm_score = self.calibrate_score(features, proposals, pdm_score)
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:87
```

Function:

```python
def calibrate_score(self, features, proposals, pdm_score):
```

Vị trí:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:98
```

Ý nghĩa:

```text
Nó lấy trajectory proposals hiện tại.
Chuyển trajectory thành simulated states.
So với past_ego_simulated_states.
Tính comfort bằng ego_is_two_frame_extended_comfort.
Calibrate lại pdm_score.
```

Công thức trong code:

```python
pdm_score[:, idx] = (14.0 * pdm_score[:, idx] + 2.0 * two_frame_comfort) / 16.0
```

Tức là:

```text
score mới = 14/16 score cũ + 2/16 comfort
```

Paper mô tả công thức dạng:

```text
S <- (7S + Sc) / 8
```

Hai cách viết tương đương tỉ lệ:

```text
14/16 = 7/8
2/16 = 1/8
```

## 18. Config Chính Của Drive-JEPA Planner

File:

```text
Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_config.py
```

Thông số mặc định:

```text
ref_num = 4
proposal_num = 32
num_poses = 8
command_num = 4
tf_d_model = 256
tf_d_ffn = 1024
tf_num_layers = 3
tf_num_head = 8
tf_dropout = 0
num_bev_layers = 1
camera_width = 1024
camera_height = 256
trajectory_sampling = time_horizon 4s, interval_length 0.5s
```

Ý nghĩa:

```text
num_poses = 8
```

Model dự đoán 8 waypoint tương lai.

```text
trajectory_sampling = 4s, 0.5s
```

8 waypoint tương ứng 4 giây tương lai, mỗi waypoint cách nhau 0.5 giây.

```text
proposal_num = 32
```

Model sinh 32 candidate trajectories.

```text
ref_num = 4
```

Refine proposals 4 vòng.

```text
tf_d_model = 256
```

Hidden dim chính của planner.

```text
tf_d_ffn = 1024
```

Hidden dim trong MLP/FFN.

```text
tf_num_head = 8
```

Số attention heads.

## 19. Training Script Của Drive-JEPA

Perception-free script:

```text
Drive-JEPA/navsim_v1/scripts/training/train_drive_jepa_perception_free.sh
```

Thông số đáng chú ý:

```text
agent.pretrain_pt_path = vitl_merge_3dataset_e50.pt
agent.image_architecture = vit_large
agent.lr = 1e-4
agent.tf_dropout = 0.0
agent.front_only = true
agent.freeze_encoder = false
agent.double_image = true
dataloader.params.batch_size = 32
trainer.params.max_epochs = 40
trainer.params.precision = bf16
```

Điểm quan trọng:

```text
Perception-free train script để freeze_encoder=false.
Tức là họ fine-tune encoder cùng planner trong script này.
```

Perception-based script:

```text
Drive-JEPA/navsim_v2/scripts/training/train_drive_jepa_perception_based.sh
```

Thông số:

```text
dataloader.params.batch_size = 32
trainer.params.max_epochs = 20
trainer.params.strategy = ddp_find_unused_parameters_true
```

Paper nói:

```text
Planner train trên 2 NVIDIA A30.
20 epochs.
Total batch size 64.
Adam.
LR planner = 1e-4.
LR ViT encoder = 1e-5.
Np = 32 proposals.
Front camera resized to 512 x 256.
```

Source code optimizer perception-based:

```python
return torch.optim.Adam(
    [
        {"params": self._pad_model._backbone.parameters(), "lr": 0.1 * self._lr},
        {"params": [p for n, p in self._pad_model.named_parameters() if "backbone" not in n], "lr": self._lr},
    ],
    lr=self._lr,
)
```

Ý nghĩa:

```text
Backbone dùng LR nhỏ hơn 10 lần.
Planner/head dùng LR chính.
Nếu lr=1e-4 thì backbone lr=1e-5.
```

## 20. So Sánh Với NN-JEPA Hiện Tại

NN-JEPA hiện tại:

```text
input:
  V-JEPA feature tokens
  state RC
  action RC

target:
  future V-JEPA feature tokens

loss:
  teacher forcing latent loss
  autoregressive rollout latent loss

output:
  latent future tokens
```

Drive-JEPA perception-free:

```text
input:
  front camera image/video
  ego status

target:
  future trajectory waypoints

loss:
  L1 trajectory loss

output:
  one trajectory [B, 8, 3]
```

Drive-JEPA perception-based:

```text
input:
  front camera image/video
  ego status
  map/geometry/meta features theo NAVSIM

target:
  human trajectory
  pseudo-teacher trajectories từ simulator
  simulator-based score labels
  auxiliary area/collision labels

loss:
  trajectory proposal loss
  score loss
  area/collision auxiliary losses

output:
  32 proposals [B, 32, 8, 3]
  score từng proposal
  selected trajectory [B, 8, 3]
```

Kết luận:

```text
NN-JEPA là latent world model.
Drive-JEPA planner là trajectory prediction/selection model.
Hai hướng có thể kết hợp, nhưng không giống nhau.
```

## 21. Nếu Áp Dụng Ý Tưởng Drive-JEPA Cho Xe RC

Với xe RC của mình, có ba hướng khả thi.

Hướng 1: Giữ NN-JEPA hiện tại.

```text
V-JEPA encoder frozen
-> feature cache
-> AC latent predictor
-> học dynamics trong latent space
```

Ưu điểm:

```text
Gần hướng world-model.
Phù hợp nếu muốn planner action bằng latent rollout.
Không cần label waypoint x,y.
```

Nhược điểm:

```text
Khó đánh giá hành vi trực tiếp.
Loss latent giảm không chắc action lái tốt hơn.
Cần planner riêng để biến latent thành action.
```

Hướng 2: Thêm Drive-JEPA style trajectory head.

```text
V-JEPA feature
state RC
-> transformer decoder/proposal head
-> dự đoán future trajectory hoặc action sequence
```

Ưu điểm:

```text
Output dễ hiểu hơn: trajectory hoặc action sequence.
Train/eval trực tiếp hơn.
Có thể log steer/throttle prediction error.
```

Nhược điểm:

```text
Cần target trajectory hoặc target action sequence tốt.
Nếu chỉ có steering/throttle thì phải đổi output từ waypoint sang action.
```

Hướng 3: Proposal-style action planner cho RC.

```text
V-JEPA feature + state
-> sinh nhiều action sequence proposals
-> scorer chọn sequence tốt nhất
```

Ví dụ output:

```text
[B, proposal_num, horizon, 2]
```

Trong đó:

```text
2 = steering, throttle
```

Ưu điểm:

```text
Gần tinh thần Drive-JEPA proposal-centric.
Có thể học multimodal behavior.
Có thể thêm scorer dự đoán an toàn/đi thẳng/không đâm nếu có rule.
```

Nhược điểm:

```text
Phức tạp hơn nhiều.
Cần metric hoặc pseudo-teacher để score proposals.
Nếu chưa có simulator/map, scorer sẽ yếu.
```

Khuyến nghị thực dụng cho RC hiện tại:

```text
Không nên nhảy ngay sang full Drive-JEPA perception-based.
Nên thử bản perception-free style nhẹ trước:
  V-JEPA feature + RC state -> Transformer decoder -> action sequence hoặc waypoint proxy.
Sau đó mới thêm proposal_num và scorer.
```

## 22. Điểm Quan Trọng Khi Không Có Waypoint Ground Truth

Drive-JEPA dự đoán:

```text
x, y, heading
```

Xe RC hiện tại đang có:

```text
steering_cmd_t
throttle_cmd_t
v_t hoặc vận tốc nếu có
yaw_rate_t
accel_x_t
accel_y_t
steering_last_t
throttle_last_t
```

Nếu không có localization hoặc odometry để suy ra trajectory `(x, y, heading)`, mình không thể copy nguyên target của Drive-JEPA.

Các lựa chọn:

```text
1. Predict action sequence:
   output [horizon, steering, throttle]

2. Predict delta state sequence:
   output [horizon, delta_yaw, delta_v, delta_accel]

3. Predict pseudo trajectory bằng dead-reckoning:
   dùng yaw_rate, velocity, dt để tích phân ra x, y, heading tương đối

4. Giữ latent world model:
   output future latent feature, sau đó planner/action head riêng
```

Nếu muốn giống Drive-JEPA nhất, cần thêm:

```text
v_t đáng tin cậy
yaw_rate_t đã sync tốt
dt giữa frame
ước lượng local trajectory tương đối
```

Sau đó có thể tạo target:

```text
future relative waypoints:
  Δx, Δy, Δheading
```

## 23. Điểm Có Thể Học Từ Drive-JEPA Cho NN-JEPA

Có bốn ý tưởng đáng port.

Ý tưởng 1: Simple transformer decoder.

```text
V-JEPA tokens
-> pooling hoặc cross-attention
-> learnable queries
-> action/trajectory sequence
```

Đây là hướng rẻ và dễ debug nhất.

Ý tưởng 2: Proposal head.

```text
Sinh nhiều candidate action sequences thay vì một action.
```

Ví dụ:

```text
proposal_num = 8 hoặc 16 trước
horizon = 4 hoặc 8
output_dim = 2
```

Ý tưởng 3: Scorer.

```text
MLP chấm điểm từng proposal.
```

Với RC, score ban đầu có thể là imitation score:

```text
proposal gần action thật nhất thì score cao.
```

Sau này nếu có collision/safety label thì bổ sung.

Ý tưởng 4: Momentum-aware selection.

```text
Khi inference, đừng chỉ chọn action sequence score cao nhất.
Phạt proposal làm steering/throttle đổi quá gắt so với bước trước.
```

Ví dụ score:

```text
score_final = model_score - λ * smoothness_penalty
```

Với xe RC, phần này rất quan trọng vì servo/throttle giật sẽ làm xe mất ổn định.

## 24. Những Gì Không Nên Copy Nguyên Xi

Không nên copy nguyên:

```text
NAVSIM metric cache
PDM/EPDM scoring
nuPlan map API
BEVFormer heavy stack
8192 anchor vocabulary
log replay traffic agents
```

Lý do:

```text
Xe RC indoor không có map chuẩn như NAVSIM.
Không có traffic agents.
Không có lane/route compliance.
Không có PDM simulator.
Copy nguyên sẽ rất nặng và không khớp dữ liệu.
```

Nên copy theo tầng:

```text
1. V-JEPA encoder feature
2. Transformer decoder với learnable queries
3. Action/trajectory sequence head
4. Proposal version nhỏ
5. Smoothness/momentum penalty
```

## 25. So Sánh Với Official V-JEPA2-AC Trong NN-JEPA

NN-JEPA official-lite hiện tại đã bám hướng:

```text
VisionTransformerPredictorAC style
state/action conditioning
token-level latent
teacher forcing loss
rollout loss
```

Drive-JEPA planner lại bám hướng:

```text
representation learning + direct planning head
```

Vì vậy không nên nói:

```text
Drive-JEPA predictor giống NN-JEPA predictor.
```

Nói đúng hơn:

```text
Drive-JEPA có V-JEPA predictor trong pretraining, nhưng planner chính không train giống NN-JEPA latent predictor.
Drive-JEPA dùng pretrained/fine-tuned V-JEPA encoder rồi train trajectory planner.
```

Nếu mục tiêu là xe tự lái thực tế, Drive-JEPA gợi ý rằng chỉ latent rollout thôi có thể chưa đủ. Cần thêm một head/action planner đọc latent và tạo hành động trực tiếp.

## 26. Kết Luận Cuối

Predictor của Drive-JEPA trong paper gồm hai tầng:

```text
Tầng self-supervised V-JEPA:
  predictor Pϕ dự đoán latent target từ context latent.

Tầng driving planner:
  transformer decoder/proposal planner dự đoán future waypoints hoặc proposals.
```

Trong source:

```text
V-JEPA AC predictor:
  Drive-JEPA/navsim_v2/vjepa2/src/models/ac_predictor.py

Perception-free trajectory predictor:
  Drive-JEPA/navsim_v1/navsim/agents/drive_jepa_perception_free/drive_jepa_model.py

Perception-based proposal planner:
  Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py
  Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py
  Drive-JEPA/navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py
```

Điểm khác biệt lớn nhất với NN-JEPA hiện tại:

```text
NN-JEPA học dự đoán latent tương lai.
Drive-JEPA dùng representation JEPA để dự đoán trajectory/proposals.
```

Hướng phát triển hợp lý cho xe RC:

```text
1. Giữ NN-JEPA latent world model đang có.
2. Thêm một action/trajectory decoder nhỏ kiểu Drive-JEPA perception-free.
3. Nếu cần multimodal, thêm proposal head nhỏ.
4. Nếu inference thật, thêm scorer và smoothness/momentum penalty.
```

