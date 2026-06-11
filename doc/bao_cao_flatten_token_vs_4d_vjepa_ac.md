# Báo cáo: Flatten token `(B, T*N, D)` và dạng 4D `(B, T, N, D)` trong V-JEPA AC

Ngày viết: 2026-06-11

## Mục tiêu

File này ghi lại rõ vì sao NN-JEPA hiện dùng latent dạng flatten `(B, T*N, D)`, trong khi repo `JEPA` của bạn trả ra/nhìn thấy dạng 4D `(B, T, N, D)`.

Kết luận ngắn:

- Source Meta `vjepa2` **có flatten** khi đưa latent vào `VisionTransformerPredictorAC`.
- Repo `JEPA` của bạn bạn cũng **có flatten**, nhưng flatten nằm bên trong model, không nằm ở Dataset.
- NN-JEPA flatten sẵn ở Dataset/train input, gần interface source Meta hơn.
- Không cần đổi NN-JEPA về return 4D trước khi train, vì không làm model mạnh hơn và có rủi ro refactor rộng.

## Ký hiệu shape

```text
B = batch size
T = số frame trong 1 sample
N = số patch token mỗi frame
D = chiều embedding latent mỗi token
```

Với V-JEPA 2.1 ViT-B 384 hiện tại của NN-JEPA:

```text
T = 8
N = 576 vì 384 / 16 = 24, 24 * 24 = 576
D = 768 vì encoder là vit_base_384
```

Do đó 1 sample latent có thể viết theo 2 dạng tương đương:

```text
Dạng 4D:
(B, 8, 576, 768)

Dạng flatten:
(B, 8*576, 768) = (B, 4608, 768)
```

Hai dạng này chứa cùng lượng thông tin nếu thứ tự token được giữ nguyên.

## Source Meta `vjepa2` làm gì?

Trong `vjepa2/app/vjepa_droid/train.py`, target encoder tạo latent rồi reshape/flatten:

```python
h = target_encoder(c)
h = h.view(batch_size, max_num_frames, -1, h.size(-1)).flatten(1, 2)
```

Ý nghĩa shape:

```text
Sau encoder:       (B*T, N, D)
view lại:          (B, T, N, D)
flatten(1, 2):     (B, T*N, D)
```

Sau đó predictor nhận `h` dạng flatten.

Trong `vjepa2/src/models/ac_predictor.py`, `VisionTransformerPredictorAC.forward()` nhận `x` dạng 3D:

```python
B, N_ctxt, D = x.size()
T = N_ctxt // (self.grid_height * self.grid_width)
x = x.view(B, T, self.grid_height * self.grid_width, D)
```

Nghĩa là predictor biết `tokens_per_frame = H*W`, nên nó có thể phục hồi lại trục frame:

```text
(B, T*N, D) -> (B, T, N, D)
```

Sau đó source Meta thêm action/state token theo từng frame:

```python
x = torch.cat([a, s, x], dim=2).flatten(1, 2)
```

Shape lúc này:

```text
(B, T, N+2, D) -> (B, T*(N+2), D)
```

Đây là chuỗi token thật sự được đưa qua Transformer.

Cuối forward, source Meta bỏ action/state token rồi flatten lại:

```python
x = x.view(B, T, cond_tokens + H*W, D)
x = x[:, :, cond_tokens:, :].flatten(1, 2)
```

Output cuối:

```text
(B, T*N, D)
```

Vì vậy source Meta dùng interface flatten cho predictor:

```text
input predictor:  (B, T*N, D)
output predictor: (B, T*N, D)
```

## Repo `JEPA` của bạn bạn làm gì?

Trong `JEPA/src/jepa_wm/data/ac_clip.py`, Dataset trả token dạng dễ đọc:

```python
z = torch.from_numpy(np.ascontiguousarray(arr[cache_rows])).float()
```

Shape:

```text
(T, N, D)
```

Sau khi DataLoader batch lại:

```text
(B, T, N, D)
```

Trong `JEPA/src/jepa_wm/models/vjepa2_ac_car.py`, model nhận 4D:

```python
def forward(self, z, a, s):
    B, T, N, D = z.shape
```

Nhưng trước khi chạy Transformer, `_embed()` vẫn flatten:

```python
x = torch.cat([at, st, zt], dim=2)
return x.flatten(1, 2)
```

Shape:

```text
(B, T, N+2, P) -> (B, T*(N+2), P)
```

Sau Transformer, model view lại 4D:

```python
x = x.view(B, T, self.group, -1)[:, :, self.cond_tokens:]
```

Output repo `JEPA`:

```text
(B, T, N, D)
```

Vì vậy repo `JEPA` không phải là không flatten. Nó chỉ đặt flatten bên trong model và return lại 4D cho dễ đọc/loss.

## NN-JEPA hiện tại làm gì?

Trong `src/data/feature_sequence_dataset.py`, Dataset đọc từng frame feature từ `.npy`, stack lại rồi flatten:

```python
latent_frames = [
    session_features.get_frame(int(sample["frame_index"]))
    for sample in sequence
]
latents = torch.stack(latent_frames, dim=0).reshape(
    self.raw_frames_per_sample * self.tokens_per_frame,
    self.embed_dim,
)
```

Shape mỗi item:

```text
(T*N, D)
```

Sau DataLoader:

```text
(B, T*N, D)
```

Với config hiện tại:

```text
(B, 8*576, 768) = (B, 4608, 768)
```

Train loop truyền thẳng dạng này vào predictor:

```python
latents = batch["latents"].to(device, non_blocking=True)
outputs = compute_world_model_losses(...)
```

NN-JEPA cắt frame bằng `tokens_per_frame`.

Ví dụ:

```text
frame 0 = latents[:, 0:576]
frame 1 = latents[:, 576:1152]
frame 2 = latents[:, 1152:1728]
...
```

Do đó dù tensor đang flatten, thông tin frame vẫn không mất.

## So sánh 3 cách biểu diễn

| Hệ thống | Dataset/training thấy gì | Model/Transformer thấy gì | Output thấy gì | Ghi chú |
|---|---:|---:|---:|---|
| Meta `vjepa2` | `(B, T*N, D)` sau train loop flatten | `(B, T*(N+cond), P)` | `(B, T*N, D)` | Predictor interface flatten |
| `JEPA` bạn bạn | `(B, T, N, D)` | `(B, T*(N+cond), P)` | `(B, T, N, D)` | Dễ đọc, flatten ẩn trong model |
| NN-JEPA | `(B, T*N, D)` | `(B, T*(N+cond), P)` | `(B, T*N, D)` | Gần interface Meta hơn |

## Vì sao Transformer cần flatten?

Transformer encoder tiêu chuẩn nhận chuỗi token:

```text
(B, L, D)
```

Nó không trực tiếp xử lý tensor 4D:

```text
(B, T, N, D)
```

Vì vậy cuối cùng kiểu gì cũng phải chuyển thành chuỗi:

```text
L = T * N
```

hoặc khi có action/state token:

```text
L = T * (N + cond_tokens)
```

Với action/state conditioning, mỗi frame có token group:

```text
[action_t, state_t, patch_1, patch_2, ..., patch_N]
```

Chuỗi Transformer là:

```text
frame 0 group, frame 1 group, frame 2 group, ...
```

Mask causal dùng group theo frame để không cho token tương lai nhìn về trước sai cách.

## Có cần đổi NN-JEPA return 4D giống `JEPA` không?

Không cần ở giai đoạn hiện tại.

Lý do:

- Source Meta `VisionTransformerPredictorAC` nhận và trả flatten `(B, T*N, D)`.
- NN-JEPA hiện đã viết loss/eval/planning quanh dạng flatten.
- Dạng flatten và 4D tương đương nếu giữ đúng `tokens_per_frame`.
- Đổi API sang 4D không làm model mạnh hơn.
- Đổi API lúc này có thể làm sai logic ở nhiều chỗ: loss, eval rollout, planning, inference, checkpoint compatibility.

Khuyến nghị hiện tại:

```text
Giữ core train/model dạng flatten.
Chỉ thêm helper view 4D khi cần debug hoặc viết metric mới.
```

Helper an toàn:

```python
def unflatten_frames(latents, tokens_per_frame):
    b, total_tokens, dim = latents.shape
    if total_tokens % tokens_per_frame != 0:
        raise ValueError("total_tokens must be divisible by tokens_per_frame")
    return latents.view(b, total_tokens // tokens_per_frame, tokens_per_frame, dim)


def flatten_frames(latents_4d):
    b, t, n, d = latents_4d.shape
    return latents_4d.view(b, t * n, d)
```

## Khi nào nên cân nhắc đổi sang API 4D?

Chỉ nên đổi nếu mục tiêu chính là readability/debug lâu dài, ví dụ:

- muốn loss đọc rõ `out[:, :-1]` vs `z[:, 1:]`;
- muốn metric theo frame/patch trực quan hơn;
- muốn code giống repo `JEPA` của bạn bạn hơn;
- sẵn sàng refactor đồng bộ train/eval/planning/inference.

Nếu đang chuẩn bị train model quan trọng, không nên đổi ngay trước train.

## Kết luận thực dụng

NN-JEPA hiện tại:

```text
(B, T*N, D)
```

là hợp lý và gần source Meta.

Repo `JEPA` của bạn bạn:

```text
(B, T, N, D)
```

dễ đọc hơn ở boundary Dataset/model, nhưng vẫn flatten bên trong trước Transformer.

Hai cách không khác bản chất. Điều quan trọng là:

- thứ tự token theo frame phải đúng;
- `tokens_per_frame` phải đúng;
- mask causal phải đúng;
- loss phải so đúng frame tương lai;
- checkpoint/feature cache không được trộn encoder khác nhau.
