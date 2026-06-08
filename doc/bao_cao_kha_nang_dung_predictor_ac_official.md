# Báo cáo: Có nên dùng `VisionTransformerPredictorAC` và `build_action_block_causal_attention_mask` cho NN-JEPA không?

Ngày viết: 2026-06-09

Repo đang xét:

- `vjepa2/`: source public V-JEPA 2/V-JEPA 2.1.
- `NN-JEPA`: repo train xe RC hiện tại.

## Kết luận ngắn

Triển khai predictor kiểu official V-JEPA-AC, tức dùng logic giống `VisionTransformerPredictorAC` và `build_action_block_causal_attention_mask`, có khả năng mạnh hơn predictor đơn giản hiện tại. Lý do là nó đưa inductive bias đúng hơn: action/state được chèn thành token điều kiện theo từng frame, patch token vẫn được giữ đầy đủ, attention mask được thiết kế theo block thời gian để hạn chế nhìn lén tương lai.

Nhưng không nên copy full config official ngay. Official config dùng predictor rất lớn, depth 24, predictor dim 1024, 16 heads. Với dữ liệu NN-JEPA hiện tại đang dùng full-token V-JEPA 2.1 384, mỗi frame có 576 token, một sample 8 frame có 4608 latent tokens, nếu thêm action/state token thì sequence dài khoảng 4624 token. Attention trên sequence dài như vậy rất nặng và dễ OOM trên GPU local 16GB.

Khuyến nghị thực dụng:

> Nên làm một bản `official-lite`: giữ ý tưởng token layout và action-block causal mask giống official, nhưng dùng predictor nhỏ trước, ví dụ depth 2 hoặc 4, predictor dim 128 hoặc 256. Train so sánh với `SimpleACPredictor` hiện tại trước khi tăng kích thước.

## Bối cảnh hiện tại

NN-JEPA hiện tại đang dùng:

- Frozen V-JEPA 2.1 encoder.
- Feature cache dạng full-token `[N, K, D]`.
- Với ViT-B/16 384: `K = 576`, `D = 768`.
- Mỗi sample train có `raw_frames_per_sample = 8`.
- Latent của một sample có shape `[8 * 576, 768] = [4608, 768]`.
- State RC hiện tại có 5 chiều.
- Action RC hiện tại có 2 chiều.
- Predictor hiện tại là `SimpleACPredictor`.
- Loss hiện tại gồm `teacher_forcing_loss + rollout_loss`.
- `auto_steps = 2`.

Source official trong `vjepa2/`:

- Predictor official nằm ở `vjepa2/src/models/ac_predictor.py`.
- Class chính là `VisionTransformerPredictorAC`.
- Attention mask official dùng `build_action_block_causal_attention_mask`.
- Train loop DROID AC nằm ở `vjepa2/app/vjepa_droid/train.py`.
- Config public AC nằm ở `vjepa2/configs/train/vitg16/droid-256px-8f.yaml`.

## Predictor hiện tại của NN-JEPA đang làm gì?

Predictor hiện tại:

```python
SimpleACPredictor
```

File:

```text
src/models/rc_jepa_ac.py
```

Luồng xử lý chính:

1. Nhận latent tokens dạng `[B, T*K, D]`.
2. Reshape thành `[B, T, K, D]`.
3. Project latent sang predictor dimension.
4. Project action/state sang token điều kiện.
5. Ghép theo từng frame:

```text
[action token, state token, patch tokens]
```

6. Flatten thành sequence dài:

```text
[B, T * (K + 2), predictor_dim]
```

7. Dùng `nn.TransformerEncoder`.
8. Dùng time-causal mask đơn giản.
9. Bỏ action/state token, chỉ lấy predicted patch tokens.
10. Project output về latent dim ban đầu.

Ưu điểm:

- Dễ đọc.
- Dễ sửa.
- Ít phụ thuộc source `vjepa2`.
- Nhẹ hơn official.
- Dễ chạy trên GPU local.
- Phù hợp để debug data, loss, W&B, resume, eval, inference.

Nhược điểm:

- Attention mask đơn giản hơn official.
- Không dùng `ACBlock`.
- Không dùng RoPE kiểu official.
- Không có action-block causal mask.
- Có thể chưa tận dụng tốt cấu trúc không gian-thời gian của token.
- Có thể yếu hơn khi dữ liệu đủ lớn và sạch.

## `VisionTransformerPredictorAC` official làm gì khác?

Class official:

```python
VisionTransformerPredictorAC
```

File:

```text
vjepa2/src/models/ac_predictor.py
```

Luồng xử lý official:

1. Nhận latent tokens:

```text
x: [B, T*K, encoder_dim]
```

2. Project latent sang predictor dimension:

```python
x = self.predictor_embed(x)
```

3. Encode state/action thành token:

```python
s = self.state_encoder(states).unsqueeze(2)
a = self.action_encoder(actions).unsqueeze(2)
```

4. Reshape latent theo frame:

```python
x = x.view(B, T, H*W, D)
```

5. Ghép action/state token vào đầu mỗi frame:

```python
x = torch.cat([a, s, x], dim=2).flatten(1, 2)
```

Nếu bật extrinsics thì layout là:

```text
[action token, state token, extrinsics token, patch tokens]
```

Nếu không bật extrinsics thì layout là:

```text
[action token, state token, patch tokens]
```

6. Chạy qua `ACBlock` với `attn_mask`:

```python
attn_mask = build_action_block_causal_attention_mask(...)
```

7. Reshape lại, bỏ token điều kiện, chỉ lấy patch tokens:

```python
x = x.view(B, T, cond_tokens + H*W, D)
x = x[:, :, cond_tokens:, :].flatten(1, 2)
```

8. Project về encoder latent dim:

```python
x = self.predictor_proj(self.predictor_norm(x))
```

Điểm quan trọng:

- Output vẫn là latent tokens.
- Predictor không dự đoán action trực tiếp.
- Predictor học dynamics trong latent space.
- Action/state chỉ là điều kiện để dự đoán latent tương lai.

## `build_action_block_causal_attention_mask` có tác dụng gì?

Mask này điều khiển token nào được phép nhìn token nào trong self-attention.

Với world model có action/state conditioning, đây là điểm rất quan trọng vì nếu mask sai, model có thể nhìn thấy thông tin tương lai trong lúc train. Khi đó train loss có thể đẹp nhưng inference thật sẽ tệ, vì lúc inference không có future latent/state để nhìn.

Về ý tưởng, mask official giúp:

- Patch tokens của frame hiện tại/quá khứ được nhìn thông tin hợp lệ.
- Token ở thời điểm tương lai không bị leak vào thời điểm quá khứ.
- Action/state token được đưa vào theo block thời gian đúng vị trí.
- Predictor học quan hệ giữa action/state và latent transition thay vì học shortcut.

So với time-causal mask đơn giản:

- Time-causal mask đơn giản thường chặn theo frame index.
- Action-block causal mask chặn chi tiết hơn theo block gồm action/state/patch tokens.
- Official mask sát layout của `VisionTransformerPredictorAC` hơn.

## Vì sao hướng official có thể mạnh hơn?

### 1. Inductive bias đúng hơn

Model được ép xử lý đúng cấu trúc:

```text
frame t = action/state điều kiện + patch tokens ảnh
```

Điều này hợp với bài toán xe RC vì action ở khoảng `t -> t+1` ảnh hưởng trực tiếp đến latent frame kế tiếp.

### 2. Giữ không gian ảnh đầy đủ

Vì không mean-pool, model vẫn thấy từng patch token. Nó có khả năng học các thay đổi không gian như:

- Mép tường dịch chuyển.
- Đường/hành lang thay đổi theo steering.
- Vật cản lớn dần khi xe tiến tới.
- Góc nhìn xoay khi xe rẽ.

Pooled feature khó học chi tiết này hơn vì mỗi frame chỉ còn một vector global.

### 3. Mask giảm nguy cơ học shortcut

Nếu attention không được chặn đúng, predictor có thể tận dụng token tương lai trong train. Official mask sinh ra để hạn chế chuyện này.

### 4. Gần source official hơn

Nếu mục tiêu nghiên cứu là bám V-JEPA-AC, predictor kiểu official giúp NN-JEPA gần paper/source hơn so với `SimpleACPredictor`.

## Tác dụng phụ và rủi ro

### 1. Rất nặng VRAM

Với NN-JEPA hiện tại:

```text
T = 8
K = 576
cond_tokens = 2
sequence_length = T * (K + cond_tokens)
                = 8 * (576 + 2)
                = 4624 tokens
```

Self-attention có chi phí gần đúng theo:

```text
O(sequence_length^2)
```

`4624^2` là hơn 21 triệu cặp attention cho mỗi head, mỗi layer, mỗi sample. Nếu batch size là 10, chi phí activation sẽ rất lớn.

Vì vậy nếu dùng predictor official lớn, OOM rất dễ xảy ra.

### 2. Official config quá lớn cho GPU local

Config official DROID AC:

```yaml
pred_depth: 24
pred_embed_dim: 1024
pred_num_heads: 16
use_activation_checkpointing: true
dtype: bfloat16
```

Đây là setup lớn. Config còn ghi:

```yaml
nodes: 4
tasks_per_node: 8
mem_per_gpu: 220G
```

Tức là official không được thiết kế cho single GPU 16GB theo kiểu train nhẹ.

### 3. Train chậm hơn nhiều

Predictor official/lite sẽ chậm hơn `SimpleACPredictor` vì:

- Mask phức tạp hơn.
- Sequence dài.
- Attention block phức tạp hơn.
- Có RoPE.
- Có nhiều projection và block riêng.

Nếu hiện tại `base` 20M đã gần 1 tiếng/epoch, bản official full có thể chậm hơn rất nhiều.

### 4. Dễ overfit

Model mạnh hơn không đồng nghĩa tốt hơn nếu data chưa đủ:

- Ít môi trường.
- Ít ánh sáng khác nhau.
- Ít tình huống rẽ/tránh/vật cản.
- Action distribution lệch.
- Sensor còn nhiễu.
- Sync camera-action chưa thật chuẩn.

Khi đó predictor mạnh có thể học thuộc trajectory thay vì học dynamics tổng quát.

### 5. Dễ implement sai layout

Đây là rủi ro lớn nhất về correctness.

Nếu dùng `build_action_block_causal_attention_mask`, layout token phải khớp đúng với mask. Ví dụ:

```text
[action, state, patch_1, patch_2, ..., patch_K]
```

Nếu code ghép token khác thứ tự, nhưng mask vẫn giả định thứ tự official, model sẽ train sai.

Các lỗi nguy hiểm:

- Action token bị đặt sai vị trí.
- State token bị đặt sai vị trí.
- Frame count `T` tính sai.
- `grid_height`, `grid_width` tính sai.
- `tokens_per_frame` không khớp `H*W`.
- Mask cho phép nhìn future patch token.
- Mask chặn nhầm action hiện tại.
- Rollout dùng state/action sai chiều thời gian.

Những lỗi này có thể không gây crash nhưng làm kết quả sai.

### 6. Khác schema DROID

Official DROID dùng:

```text
actions: [B, T-1, 7]
states:  [B, T,   7]
```

NN-JEPA dùng:

```text
actions: [B, T-1, 2]
states:  [B, T,   5]
```

Nếu copy `VisionTransformerPredictorAC` nguyên xi, cần sửa:

- `action_embed_dim`.
- `state_embed_dim`.
- Có thể tách `action_embed_dim` và `state_embed_dim`, vì official mặc định dùng cùng một dim.
- Không dùng extrinsics nếu không có dữ liệu camera extrinsics.

### 7. Có thể cần mixed precision

Official dùng `bfloat16`. NN-JEPA hiện tại ưu tiên `fp32` feature cache để giữ chính xác, nhưng train predictor full-token có thể cần:

- `bfloat16` autocast.
- Activation checkpointing.
- Batch size nhỏ.
- Eval batch size rất nhỏ.
- Gradient accumulation nếu muốn batch effective lớn hơn.

Nếu giữ hoàn toàn `fp32`, bản official-like có thể quá nặng.

## Nên triển khai theo cách nào?

Không nên thay `SimpleACPredictor` trực tiếp bằng full official predictor ngay. Nên triển khai theo từng mức:

### Mức 1: Official-lite predictor

Mục tiêu:

- Giữ layout official.
- Giữ action/state token block.
- Dùng causal mask kiểu official.
- Dùng model nhỏ để chạy được.

Preset đề xuất:

```text
official_tiny:
  predictor_dim = 128
  depth = 2
  heads = 4
```

```text
official_small:
  predictor_dim = 256
  depth = 4
  heads = 4
```

Không bật extrinsics.

Không dùng depth 24 ngay.

### Mức 2: So sánh với SimpleACPredictor

Chạy cùng:

- Cùng data split.
- Cùng feature cache.
- Cùng `raw_frames_per_sample = 8`.
- Cùng `auto_steps = 2`.
- Cùng optimizer/lr/scheduler.
- Cùng seed.
- Cùng eval/test script.

So sánh:

- `val/loss`.
- `val/teacher_forcing_loss`.
- `val/rollout_loss`.
- `test/loss`.
- Inference rollout qualitative.
- Tốc độ train mỗi epoch.
- VRAM peak.

### Mức 3: Tăng kích thước nếu thật sự tốt hơn

Nếu official-lite tốt hơn ổn định:

- Tăng depth lên 6 hoặc 8.
- Tăng predictor dim lên 384 hoặc 512.
- Giữ heads sao cho `predictor_dim % heads == 0`.
- Theo dõi OOM và overfit.

Chỉ nên thử model lớn hơn khi:

- Data cache đủ.
- GPU ổn định.
- Train/eval không OOM.
- W&B log đầy đủ.
- Loss trên val/test cải thiện thật.

## Kế hoạch triển khai an toàn

### Bước 1: Không sửa predictor cũ

Giữ `SimpleACPredictor` làm baseline. Thêm class mới, ví dụ:

```python
OfficialLiteACPredictor
```

hoặc:

```python
VJepaStyleACPredictor
```

Mục tiêu là có thể chọn bằng CLI:

```bash
--predictor-type simple
--predictor-type official_lite
```

### Bước 2: Viết unit test cho mask và shape

Test cần có:

- Input latent `[B, T*K, D]`.
- Action `[B, T-1, A]` hoặc `[B, T, A]` tùy convention nội bộ.
- State `[B, T, S]`.
- Output phải là `[B, T*K, D]` hoặc `[B, (T-1)*K, D]` tùy nhánh prediction.
- Mask không cho frame quá khứ nhìn frame tương lai.
- Token layout phải đúng `[action, state, patches]`.

### Bước 3: Smoke test CPU/GPU batch nhỏ

Chạy batch size 1 hoặc 2 trước:

```bash
--model-size official_tiny
--batch-size 1
--eval-batch-size 1
--epochs 1
```

Chỉ tăng batch sau khi chắc chắn không OOM.

### Bước 4: So sánh chính thức với baseline

Chạy:

- `simple_tiny`.
- `official_tiny`.
- `simple_small`.
- `official_small`.

Nếu official-lite không thắng `simple` trên val/test thì chưa nên tăng size.

## Dấu hiệu cho thấy official-like predictor thật sự có ích

Nên xem các tín hiệu sau:

- `val/rollout_loss` giảm tốt hơn `simple`.
- `test/rollout_loss` giảm tốt hơn, không chỉ train loss.
- Rollout dài hơn không bị drift quá nhanh.
- Inference qualitative nhìn hợp lý hơn khi steering/throttle thay đổi.
- Không bị overfit: train loss giảm nhưng val/test không tăng.
- Tốc độ train vẫn chấp nhận được.

Nếu chỉ `train/loss` giảm nhưng `val/test` không cải thiện, có thể predictor mạnh đang học thuộc data.

## Dấu hiệu nên dừng hướng official-like

Nên tạm dừng nếu:

- Batch size 1 vẫn OOM.
- Train chậm đến mức không thể thử nghiệm.
- Loss không tốt hơn simple sau nhiều run.
- Implementation mask quá khó kiểm chứng.
- Rollout inference tệ hơn dù val loss giảm.
- Data hiện tại chưa đủ sạch hoặc feature cache còn thiếu session.

## Có nên dùng checkpoint AC official không?

Checkpoint public:

```text
vjepa2-ac-vitg.pt
```

Nó là V-JEPA 2-AC từ ViT-g/16, ảnh 256. NN-JEPA hiện tại đang dùng V-JEPA 2.1 ViT-B/16 384. Hai hệ này không khớp trực tiếp:

- Encoder khác.
- Resolution khác.
- Embed dim khác.
- Predictor architecture/size khác.
- State/action schema khác.

Vì vậy không nên kỳ vọng load predictor official AC vào NN-JEPA hiện tại một cách trực tiếp. Nếu muốn dùng checkpoint official AC, cần một nhánh riêng:

- Dùng encoder ViT-g/16 256 tương ứng.
- Dùng predictor architecture giống official.
- Adapt state/action RC từ 2D/5D sang schema phù hợp.
- Chấp nhận chi phí VRAM rất lớn.

Trong giai đoạn hiện tại, cách hợp lý hơn là học predictor mới trên data RC.

## Kết luận cuối

Dùng `VisionTransformerPredictorAC` và `build_action_block_causal_attention_mask` có thể giúp NN-JEPA mạnh hơn và gần official V-JEPA-AC hơn. Nhưng lợi ích này đi kèm chi phí lớn: VRAM cao, train chậm, dễ OOM, dễ overfit, và đặc biệt dễ sai nếu token layout/mask không khớp.

Hướng nên làm là không thay thế ngay predictor hiện tại. Hãy thêm một nhánh `official-lite`, giữ `SimpleACPredictor` làm baseline, rồi so sánh nghiêm túc trên val/test. Nếu `official-lite` thắng rõ ràng và ổn định, mới tăng kích thước predictor. Nếu không thắng, predictor đơn giản hiện tại vẫn là lựa chọn tốt hơn cho giai đoạn xây pipeline, kiểm tra data, và chạy thử nghiệm nhanh trên xe RC.
