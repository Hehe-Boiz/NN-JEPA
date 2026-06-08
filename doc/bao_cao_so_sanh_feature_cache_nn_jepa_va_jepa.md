# Báo Cáo So Sánh Feature Cache Giữa NN-JEPA Và JEPA

Ngày viết: 2026-06-09

## 1. Kết luận ngắn

Lý do feature cache của NN-JEPA đang khoảng `145 GiB` dù chỉ dùng `vitb_384`, còn repo `JEPA` của bạn có thể nhỏ hơn rất nhiều dù dùng `ViT-L`, là vì hai repo đang lưu **hai loại latent khác nhau**.

```text
NN-JEPA hiện tại:
  lưu toàn bộ spatial patch tokens mỗi frame
  shape mỗi frame = (576, D)

JEPA:
  mean-pool 576 spatial tokens thành 1 vector duy nhất mỗi frame
  shape mỗi frame = (D,)
```

Vì `576` token/frame bị giữ nguyên trong NN-JEPA, dung lượng tăng đúng khoảng `576 lần` so với pooled latent nếu cùng `D` và cùng dtype.

## 2. JEPA đang làm gì?

Trong repo `JEPA`, file:

```text
JEPA/src/jepa_wm/engine/encode.py
```

Docstring ghi rõ pipeline:

```text
frame (B,3,1,384,384) -> encoder -> (B, 576, 1024) spatial tokens -> mean-pool -> (B, 1024)
```

Code thực tế:

```python
tok = enc(x)                         # (B, 576, 1024)
lats.append(tok.float().mean(1).cpu())  # (B, 1024)
```

Nghĩa là `JEPA` dùng encoder `vjepa2_1_vit_large_384`, encoder trả về `576` patch token cho mỗi frame, nhưng sau đó lấy trung bình theo chiều token:

```text
(576, 1024) -> mean over 576 -> (1024,)
```

Sau đó repo `JEPA` lưu từng session dạng:

```text
data/latents/<session>.pt = {
  "latents": FloatTensor (N, 1024),
  "frame_idx": LongTensor (N,)
}
```

Dataset trong:

```text
JEPA/src/jepa_wm/data/dataset.py
```

cũng xác nhận nó đọc latent dạng:

```text
{"latents": FloatTensor (N, D), "frame_idx": LongTensor (N,)}
```

## 3. NN-JEPA hiện tại đang làm gì?

NN-JEPA hiện tại trong:

```text
src/tools/extract_vjepa_features.py
```

lưu feature mỗi session dạng `.npy` với layout:

```text
feature_layout = frame_tokens
shape = (num_frames, tokens_per_frame, embed_dim)
```

Với `vitb_384`:

```text
tokens_per_frame = 576
embed_dim = 768
dtype = fp32
shape mỗi frame = (576, 768)
```

Nên mỗi frame cần:

```text
576 * 768 * 4 bytes = 1,769,472 bytes ~= 1.6875 MiB/frame
```

Với data hiện tại:

```text
frame_count = 88,120
```

dung lượng lý thuyết:

```text
88,120 * 576 * 768 * 4 bytes ~= 145.22 GiB
```

Kết quả thực tế trên máy:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp32 ~= 146 GiB
```

Con số này khớp với công thức, nên đây không phải lỗi ghi file.

## 4. Vì sao bạn dùng ViT-B mà vẫn nặng hơn ViT-L của JEPA?

Vì chiều lưu khác nhau:

```text
NN-JEPA ViT-B full token:
  mỗi frame = 576 * 768 float32

JEPA ViT-L pooled:
  mỗi frame = 1024 float32
```

So sánh số float mỗi frame:

```text
NN-JEPA ViT-B full token = 576 * 768 = 442,368 float/frame
JEPA ViT-L pooled        = 1 * 1024  = 1,024 float/frame
```

Tỉ lệ:

```text
442,368 / 1,024 = 432 lần
```

Nên dù `ViT-L` mạnh hơn `ViT-B`, pooled latent của `JEPA` vẫn nhỏ hơn rất nhiều.

## 5. Ước tính dung lượng nếu NN-JEPA cũng dùng pooled latent

Với `88,120` frame và `fp32`:

```text
vitb_384 pooled:
  88,120 * 768 * 4 bytes ~= 258 MiB

vitl_384 pooled:
  88,120 * 1024 * 4 bytes ~= 344 MiB

vitg_384 pooled:
  88,120 * 1408 * 4 bytes ~= 474 MiB

vitG_384 pooled:
  88,120 * 1664 * 4 bytes ~= 559 MiB
```

Nếu dùng `fp16`, các con số trên giảm một nửa.

Vì vậy nếu chỉ lưu pooled latent, cache không thể lên `60 GiB` với số frame hiện tại. Nếu bạn thấy folder của bạn khoảng `60 GiB`, khả năng cao đó là tổng nhiều thứ khác như raw frames, video, processed images, zip, hoặc cache trung gian khác, không phải riêng pooled latent.

## 6. Ước tính dung lượng full-token hiện tại

Với `88,120` frame:

```text
vitb_384 full token fp32:
  576 * 768 * 4 bytes/frame ~= 145.22 GiB

vitl_384 full token fp32:
  576 * 1024 * 4 bytes/frame ~= 193.62 GiB

vitg_384 full token fp32:
  576 * 1408 * 4 bytes/frame ~= 266.23 GiB

vitG_384 full token fp32:
  576 * 1664 * 4 bytes/frame ~= 314.64 GiB
```

Nếu dùng `fp16`, các con số này giảm một nửa:

```text
vitb_384 full token fp16 ~= 72.61 GiB
vitl_384 full token fp16 ~= 96.81 GiB
vitg_384 full token fp16 ~= 133.12 GiB
vitG_384 full token fp16 ~= 157.32 GiB
```

## 7. Tradeoff: full-token latent vs pooled latent

### Full-token latent

Ưu điểm:

- Giữ thông tin spatial patch đầy đủ.
- Gần hơn với cách JEPA/V-JEPA biểu diễn feature dạng token.
- Predictor có thể học thay đổi theo vùng ảnh, ví dụ vật thể/đường/biên tường ở vị trí khác nhau.

Nhược điểm:

- Rất nặng disk.
- Train predictor chậm và dễ OOM.
- Một sample `8 frame` có:

```text
8 * 576 = 4,608 token
```

Batch `10` là khoảng:

```text
46,080 token
```

trước khi đi qua transformer predictor.

### Pooled latent

Ưu điểm:

- Rất nhẹ.
- Train nhanh.
- Predictor đơn giản hơn nhiều.
- Repo `JEPA` đã chứng minh workflow này chạy được cho `vjepa_ac`.

Nhược điểm:

- Mất thông tin vị trí/spatial patch.
- Chỉ còn global visual descriptor của frame.
- Có thể yếu hơn nếu bài toán cần biết vật thể/đường nằm ở đâu trong ảnh.

## 8. Điểm quan trọng về control

Trong `JEPA/docs/HANDOFF.md`, có một nhận xét rất quan trọng:

```text
Multi-frame clip (feed T=4/8) = nâng cấp chính cho CONTROL.
Mạnh hơn đổi ViT-G.
Encoder không phải bottleneck.
```

Ý nghĩa:

- Với xe RC, chỉ dùng 1 frame tĩnh thì latent không biết vận tốc/chuyển động.
- Đổi từ ViT-L sang ViT-G có thể không cải thiện bằng việc encode multi-frame clip.
- Nếu mục tiêu là điều khiển xe, motion information có thể quan trọng hơn model size.

Hiện NN-JEPA đang dùng sequence `8 frame`, nhưng mỗi frame được encode độc lập thành pseudo-clip `tubelet_size=2`. Nghĩa là predictor thấy chuỗi latent theo thời gian, nhưng encoder từng frame không thật sự encode motion trong một clip nhiều frame.

## 9. Kết luận kỹ thuật

Không có lỗi dung lượng trong NN-JEPA hiện tại. Dung lượng `~145 GiB` là đúng với thiết kế:

```text
frame_tokens + fp32 + vitb_384 + 88,120 frame
```

Repo `JEPA` nhỏ hơn vì dùng:

```text
mean-pooled latent (N, D)
```

chứ không lưu:

```text
(N, 576, D)
```

Nếu muốn giống bạn của bạn, NN-JEPA cần thêm một pipeline mới:

```text
--feature-layout pooled
```

và thêm model/dataset train tương ứng cho pooled latent. Không nên chỉ đổi extractor rồi dùng lại predictor hiện tại, vì predictor hiện tại đang thiết kế cho token-level latent.

## 10. Đề xuất thực tế

Thứ tự nên làm:

1. Giữ cache full-token `vitb_384 fp32` hiện tại nếu muốn bám sát token-level JEPA.
2. Thêm nhánh thử nghiệm `pooled` giống repo `JEPA` để train nhanh và benchmark.
3. So sánh `pooled ViT-L` với `full-token ViT-B` bằng cùng data split và cùng metric val/test.
4. Nếu pooled hoạt động tốt, dùng pooled để chạy nhiều experiment nhanh.
5. Nếu cần chất lượng cao hơn, thử multi-frame clip trước khi nhảy lên ViT-G/ViT-Gigantic.

