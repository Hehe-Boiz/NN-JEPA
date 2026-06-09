# Báo cáo: vì sao `train batch_size=32` không OOM nhưng `eval_batch_size=16` lại OOM

Ngày viết: 2026-06-09

Repo: `NN-JEPA`

## Tóm tắt ngắn

Hiện tượng này không mâu thuẫn. `eval_batch_size=16` nhỏ hơn `train.batch_size=32`, nhưng val/eval vẫn có thể OOM vì eval không nhất thiết dùng cùng kernel/bộ nhớ với train.

Trong code hiện tại có 3 nguyên nhân chính:

1. `eval` chạy trong `model.eval()` + `torch.no_grad()`, nên PyTorch có thể đi vào native fastpath riêng của `nn.TransformerEncoderLayer`.
2. Trace OOM trước đó rơi đúng vào `torch._transformer_encoder_layer_fwd`, tức là đường eval fastpath, không phải đường train bình thường.
3. Val chạy ngay sau train epoch. Nếu không clear gradient trước val, tensor `parameter.grad` của batch train cuối vẫn nằm trên GPU, làm val bắt đầu trong trạng thái VRAM chưa sạch.

Vì vậy câu trả lời ngắn là:

```text
train batch 32 không OOM
không đảm bảo
eval batch 16 cũng không OOM
```

Đặc biệt với sequence rất dài như pipeline hiện tại.

## Kích thước tensor hiện tại

Feature cache hiện tại lấy từ V-JEPA 2.1 ViT-B/16 384.

Metadata quan trọng:

```text
image_size = 384
patch_size = 16
tokens_per_frame = 24 * 24 = 576
embed_dim = 768
raw_frames_per_sample = 8
auto_steps = 2
```

Một sample train có:

```text
latents = [8 frame * 576 token/frame, 768]
        = [4608, 768]
```

Trong `SimpleACPredictor`, mỗi frame được thêm:

```text
1 action token
1 state token
576 latent patch tokens
```

Nên sequence thật sự đi vào Transformer là:

```text
tokens_per_step = 576 + 2 = 578
sequence_length = 8 * 578 = 4624 token
```

Đây là sequence rất dài đối với self-attention.

## Vì sao attention dễ OOM

Self-attention có memory tăng theo gần:

```text
batch_size * num_heads * sequence_length^2
```

Với tiny hiện tại:

```text
num_heads = 4
sequence_length = 4624
```

Riêng attention score lý thuyết đã có kích thước:

```text
B * 4 * 4624 * 4624
```

Nếu `B=16`:

```text
16 * 4 * 4624 * 4624 ~= 1.37 tỷ phần tử
```

Nếu tensor này ở `float32`, chỉ riêng nó đã khoảng:

```text
1.37e9 * 4 bytes ~= 5.5 GB
```

Đó mới là một phần của memory. Còn có:

```text
Q/K/V
activation trung gian
output attention
MLP hidden
mask
parameter
optimizer state
gradient còn sót sau train
CUDA allocator reserve/cache
W&B watch/gradient logging nếu đang bật
```

Do đó `eval_batch_size=16` vẫn có thể vượt VRAM 16 GB.

## Vì sao train batch 32 có thể không OOM

Nghe có vẻ ngược, vì train thường tốn memory hơn eval. Nhưng trường hợp này khác vì train và eval có thể không dùng cùng đường thực thi trong PyTorch.

### 1. Train không đi vào native eval fastpath

Khi train:

```python
predictor.train(True)
torch.set_grad_enabled(True)
```

PyTorch sẽ dùng đường forward/backward bình thường.

Khi val:

```python
predictor.train(False)
torch.no_grad()
```

`nn.TransformerEncoderLayer` có thể kích hoạt native fastpath tối ưu riêng.

Trace OOM bạn gửi trước đó có đoạn:

```text
torch._transformer_encoder_layer_fwd
torch.OutOfMemoryError: Tried to allocate 4.88 GiB
```

Điểm này rất quan trọng. Nó cho thấy OOM xảy ra trong native fused Transformer eval path, không phải trong code Python predictor thông thường.

### 2. Native fastpath không chắc tiết kiệm VRAM hơn

Fastpath thường tối ưu tốc độ, nhưng không đảm bảo peak VRAM thấp hơn với mọi shape.

Với sequence dài `4624`, fastpath có thể tạo workspace hoặc attention buffer lớn hơn đường train thông thường.

Vì vậy:

```text
eval batch nhỏ hơn train
nhưng eval kernel peak memory lớn hơn
```

là chuyện có thể xảy ra.

### 3. Val chạy ngay sau train, GPU chưa thật sự sạch

Trong PyTorch, sau batch train cuối:

```python
loss.backward()
optimizer.step()
```

Gradient trong `parameter.grad` vẫn còn tồn tại cho tới khi gọi:

```python
optimizer.zero_grad(set_to_none=True)
```

Trong train loop cũ, sau train epoch xong là chạy val ngay. Nếu chưa clear gradient, val sẽ cộng thêm memory đang giữ trong `parameter.grad`.

Điều này không làm sai kết quả val, nhưng làm tăng VRAM peak.

## Code cũ có sai không?

Không sai theo nghĩa logic train/eval.

Các điểm đúng:

```text
val dùng dataloader["val"]
val dùng eval_batch_size riêng
val chạy optimizer=None
val không backward
val nằm trong torch.no_grad()
model được đặt predictor.train(False)
```

Nhưng code cũ chưa đủ an toàn cho GPU memory với sequence dài.

Cụ thể có 2 điểm cần cải thiện:

1. Chưa tắt eval native fastpath của `nn.TransformerEncoderLayer`.
2. Chưa clear gradient trước khi vào val/test.

Hai điểm này không phải lỗi toán học, nhưng là lỗi thực dụng về memory khi train sequence dài.

## Những thay đổi đã thực hiện

### 1. Tắt eval fastpath cho `SimpleACPredictor`

File:

```text
src/models/rc_jepa_ac.py
```

Thêm context manager:

```python
@contextmanager
def torch_transformer_eval_fastpath_disabled(disable: bool):
    """Avoid eval-only native Transformer fastpath memory spikes on long token sequences."""
    ...
```

Vị trí hiện tại:

```text
src/models/rc_jepa_ac.py:49
```

Sau đó bọc đoạn gọi Transformer:

```python
with torch_transformer_eval_fastpath_disabled(disable=not self.training):
    sequence = self.blocks(sequence, mask=mask)
```

Vị trí hiện tại:

```text
src/models/rc_jepa_ac.py:335
```

Ý nghĩa:

```text
train mode: không thay đổi
eval mode: tắt native MHA/Transformer fastpath tạm thời
sau forward: restore lại trạng thái fastpath ban đầu
```

Điểm quan trọng: thay đổi này chỉ ảnh hưởng `SimpleACPredictor` khi `eval`. Nó không đổi kiến trúc, không đổi loss, không đổi dữ liệu, không đổi kết quả theo nghĩa học mô hình. Nó chỉ ép PyTorch dùng path ít rủi ro OOM hơn.

### 2. Clear gradient trước val trong feature-cache train

File:

```text
src/tools/train_rc_jepa_ac_features.py
```

Đã thêm:

```python
optimizer.zero_grad(set_to_none=True)
maybe_cleanup_cuda()
```

ngay trước val.

Vị trí hiện tại:

```text
src/tools/train_rc_jepa_ac_features.py:527
```

Mục đích:

```text
xóa parameter.grad còn sót từ batch train cuối
cho CUDA allocator có cơ hội giải phóng cache không cần thiết
giảm VRAM trước khi vào val
```

### 3. Clear gradient trước test trong feature-cache train

File:

```text
src/tools/train_rc_jepa_ac_features.py
```

Đã thêm:

```python
optimizer.zero_grad(set_to_none=True)
maybe_cleanup_cuda()
```

trước test cuối cùng.

Vị trí hiện tại:

```text
src/tools/train_rc_jepa_ac_features.py:639
```

Mục đích tương tự val.

### 4. Clear gradient trước val/test trong raw-frame train

File:

```text
src/tools/train_rc_jepa_ac.py
```

Đã thêm clear gradient trước val:

```text
src/tools/train_rc_jepa_ac.py:585
```

Đã thêm clear gradient trước test:

```text
src/tools/train_rc_jepa_ac.py:710
```

Mục đích: giữ hai path train đồng nhất. Dù hiện tại mình chủ yếu train từ feature cache, raw-frame path cũng nên an toàn.

### 5. Giảm `eval_batch_size` về 2 cho tiny newdata

File:

```text
configs/hydra/experiment/rc_jepa_tiny_newdata.yaml
```

Hiện tại:

```yaml
train:
  epochs: 100
  batch_size: 32
  eval_batch_size: 2
```

Vị trí hiện tại:

```text
configs/hydra/experiment/rc_jepa_tiny_newdata.yaml:33
```

Lý do:

```text
eval_batch_size=16 quá rủi ro với sequence_length=4624
eval_batch_size=2 an toàn hơn cho val/test qua đêm
train batch_size=32 có thể giữ nếu GPU chịu được
```

## Vì sao không giảm luôn train batch size?

Vì lỗi bạn hỏi xảy ra ở val/eval, không phải train.

Nếu train batch 32 đang chạy ổn thì giữ lại có lợi:

```text
tận dụng GPU tốt hơn
ít step hơn mỗi epoch
train nhanh hơn
```

Nhưng nếu train cũng OOM, lúc đó mới giảm:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.batch_size=16
```

Nếu vẫn OOM:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.batch_size=8
```

## Vì sao không dùng `eval_batch_size=8`?

Có thể dùng được, nhưng hiện tại mục tiêu là chạy qua đêm không debug.

`eval_batch_size=2` là lựa chọn an toàn hơn vì:

```text
val/test không ảnh hưởng gradient
eval batch nhỏ chỉ làm val/test chậm hơn
không làm giảm chất lượng model
không thay đổi train update
giảm mạnh nguy cơ OOM
```

Với val:

```text
val_windows = 8259
eval_batch_size=2 -> khoảng 4130 batch val
```

Val sẽ chậm hơn, nhưng an toàn hơn. Sau khi train ổn định, có thể benchmark lại `eval_batch_size=4` hoặc `8`.

## `eval_batch_size` có ảnh hưởng chất lượng model không?

Không ảnh hưởng trực tiếp.

Trong val/test:

```text
không backward
không optimizer.step()
không update weight
```

Nó chỉ ảnh hưởng:

```text
tốc độ eval
VRAM eval
độ mượt progress bar
```

Loss val được cộng trung bình theo số sample:

```python
totals[key] += outputs[key] * batch_size
average = total / total_samples
```

Nên nếu code deterministic và không có dropout ở eval, `eval_batch_size=2` hay `16` về lý thuyết cho cùng metric gần như giống nhau, chỉ khác sai số floating point rất nhỏ.

## Train batch size có ảnh hưởng chất lượng model không?

Có thể có.

Train batch size ảnh hưởng:

```text
gradient estimate
optimizer update
learning dynamics
noise của gradient
```

Do đó train batch không giống eval batch. Giảm eval batch là an toàn hơn giảm train batch nếu mục tiêu chỉ là tránh OOM khi val.

## Trạng thái config hiện tại

Config:

```text
configs/hydra/experiment/rc_jepa_tiny_newdata.yaml
```

Hiện tại là:

```yaml
model:
  type: simple
  size: tiny

train:
  epochs: 100
  batch_size: 32
  eval_batch_size: 2
  num_workers: 8
  lr: 1.0e-4
  weight_decay: 1.0e-4
  grad_clip: 1.0
  warmup_epochs: 5
  warmup_start_factor: 0.1
  min_lr_ratio: 0.1
  early_stopping_patience: 15
```

Dry-run đã xác nhận:

```text
predictor_type = simple
model_size = tiny
predictor_dim = 128
predictor_depth = 2
predictor_heads = 4
batch_size = 32
eval_batch_size = 2
raw_frames_per_sample = 8
auto_steps = 2
output_dir = checkpoints/rc_jepa_ac_vitb_features_newdata_tiny
resume_from = null
```

## Lệnh train hiện tại

Lệnh chính:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra experiment=rc_jepa_tiny_newdata
```

Nếu muốn ép team W&B:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  wandb.entity=TEN_TEAM_CUA_BAN
```

Nếu train batch 32 cũng OOM:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.batch_size=16
```

Nếu muốn test eval batch lớn hơn sau khi ổn định:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  train.eval_batch_size=4
```

Không khuyến nghị tăng eval batch ngay trước khi ngủ.

## Kiểm tra đã chạy sau khi sửa

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m compileall -q src
```

Kết quả:

```text
pass
```

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m unittest discover -s tests
```

Kết quả:

```text
Ran 41 tests
OK
```

Đã chạy dry-run:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata \
  runtime.dry_run=true \
  runtime.require_cuda=false \
  wandb.disabled=true \
  train.device=cpu
```

Kết quả xác nhận:

```text
batch_size = 32
eval_batch_size = 2
predictor_type = simple
model_size = tiny
resume_from = null
```

## Kết luận

Lý do `train batch 32` có thể chạy nhưng `eval batch 16` lại OOM là do eval path của PyTorch không giống train path, đặc biệt với `nn.TransformerEncoderLayer` trên sequence dài `4624 token`.

Thay đổi hiện tại xử lý theo 3 lớp an toàn:

```text
tắt eval native fastpath gây peak memory bất thường
clear gradient trước val/test
giảm eval_batch_size về 2
```

Đây là hướng đúng cho giai đoạn chạy qua đêm: ưu tiên không OOM, không làm đổi chất lượng train, không đổi data, không đổi kiến trúc model.

