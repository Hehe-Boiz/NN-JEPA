# Báo Cáo So Sánh JEPA Của Bạn Bạn Và NN-JEPA Hiện Tại

Ngày kiểm tra: 2026-06-09

Mục tiêu của báo cáo này là trả lời câu hỏi: vì sao repo `JEPA` mới của bạn bạn train V-JEPA 2.1 AC ổn định RAM/VRAM hơn NN-JEPA hiện tại, dù model predictor của họ có vẻ nhiều tham số hơn.

Kết luận ngắn: sự ổn định không đến từ số tham số predictor. Nó đến từ thiết kế pipeline: feature cache `fp16`, train/eval bằng `bfloat16`, sample ngắn hơn, batch theo session, DataLoader ít prefetch/persistent hơn, và W&B logging nhẹ hơn. NN-JEPA hiện tại có thể dùng predictor tiny rất ít tham số nhưng vẫn OOM vì mỗi sample latent quá dài và DataLoader prefetch quá nhiều dữ liệu `fp32`.

---

## 1. Repo JEPA Hiện Tại Đang Ở Trạng Thái Nào?

Repo `JEPA` đang ở commit:

```text
b4bb9e4 Retrain prep: 384px control encode + prev-action state + depth-12 (reviewed vs Meta)
```

Commit này cập nhật pipeline AC car theo hướng:

- Dùng V-JEPA 2.1 ở ảnh `384px`.
- Patch token mỗi frame tăng từ `256` token lên `576` token.
- State tăng từ `10D` lên `12D` nhờ thêm `prev_steer`, `prev_throttle`.
- Predictor depth tăng từ `8` lên `12`.
- Batch size giảm từ `40` xuống `24` để tránh nặng VRAM khi dùng 576 token/frame.
- Encode patch cache mặc định sang `data/latents_towerpro_patch_384`.
- Train vẫn dùng frozen V-JEPA encoder, tức encoder không nằm trong graph train predictor.

Các file chính của JEPA:

- `JEPA/configs/model/vjepa_ac_car.yaml`
- `JEPA/configs/train/vjepa_ac_car.yaml`
- `JEPA/src/jepa_wm/engine/encode_patch.py`
- `JEPA/src/jepa_wm/data/ac_clip.py`
- `JEPA/src/jepa_wm/engine/train_ac_car.py`
- `JEPA/src/jepa_wm/models/vjepa2_ac_car.py`

---

## 2. Số Tham Số Thực Tế

Đã đếm trực tiếp bằng code model, không ước lượng bằng tay.

### 2.1. JEPA

| Model trong JEPA | Cấu hình chính | Số tham số |
|---|---:|---:|
| `vjepa_ac_car.yaml` | 384px, 576 token/frame, D=1024, pred_dim=512, depth=12 | `39.19M` |
| `vjepa_ac_car_minimal.yaml` | 256px, 256 token/frame, D=1024, pred_dim=512, depth=8 | `26.41M` |
| `vjepa_ac.yaml` | pooled latent baseline | `7.36M` |

Điểm quan trọng: bản bạn nói khoảng `26M` nhiều khả năng là `vjepa_ac_car_minimal`, không phải bản mới nhất `vjepa_ac_car` ở commit hiện tại. Bản mới nhất đang là khoảng `39M`.

### 2.2. NN-JEPA

NN-JEPA hiện tại với feature cache `vjepa2_1_vitb_384_ema_fp32`, tức encoder ViT-B, D=768, 576 token/frame:

| Predictor NN-JEPA | Cấu hình | Số tham số |
|---|---:|---:|
| `simple tiny` | dim=128, depth=2, heads=4 | `0.67M` |
| `simple small` | dim=256, depth=4, heads=4 | `3.71M` |
| `simple base` | dim=512, depth=6, heads=8 | `20.01M` |
| `official_lite tiny` | dim=128, depth=2, heads=4 | `0.60M` |
| `official_lite small` | dim=256, depth=4, heads=4 | `3.56M` |
| `official_lite base` | dim=512, depth=6, heads=8 | `19.71M` |

Kết luận: NN-JEPA tiny nhỏ hơn JEPA rất nhiều, nhưng vẫn có thể bị RAM OOM. Vì vậy thủ phạm chính không phải số tham số predictor.

---

## 3. Khác Biệt Encoder Và Feature Cache

## 3.0. `256px` Và `384px` Là Gì?

`256px` hoặc `384px` trong các config V-JEPA là kích thước ảnh đầu vào sau resize trước khi đưa vào encoder. Vì V-JEPA dùng ảnh vuông, nên có thể hiểu là:

```text
256px = ảnh 256 x 256
384px = ảnh 384 x 384
```

Kích thước này quyết định trực tiếp số patch token mỗi frame. Với patch size `16`:

```text
256 / 16 = 16 patch mỗi chiều
16 x 16 = 256 token/frame
```

```text
384 / 16 = 24 patch mỗi chiều
24 x 24 = 576 token/frame
```

Vì vậy khi đổi từ `256px` lên `384px`, số token mỗi frame tăng:

```text
576 / 256 = 2.25 lần
```

Nhưng chi phí attention không tăng tuyến tính đơn giản, vì attention phụ thuộc mạnh vào bình phương sequence length. Do đó `384px` có thể tốt hơn về độ chi tiết ảnh, nhưng nặng RAM/VRAM hơn đáng kể.

### NN-JEPA hiện tại đang dùng kích thước ảnh bao nhiêu?

NN-JEPA hiện tại, đối với pipeline V-JEPA AC đang train từ feature cache, dùng ảnh `384 x 384`.

Bằng chứng trong code/config:

```python
AC_IMAGE_SIZE = 384
```

Feature cache hiện tại:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

Metadata của feature cache:

```json
{
  "encoder_preset": "vitb_384",
  "encoder_name": "vit_base_384",
  "image_size": 384,
  "patch_size": 16,
  "tokens_per_frame": 576,
  "embed_dim": 768,
  "dtype": "fp32"
}
```

Điểm dễ nhầm: trong `src/data/settings.py` vẫn còn:

```python
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
```

Hai giá trị `224` này thuộc pipeline preprocess ảnh/CNN cũ, không phải kích thước đang dùng cho V-JEPA AC feature-cache hiện tại. Với train hiện tại bằng:

```text
features_dir: data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

thì kích thước ảnh đã dùng khi extract feature là `384 x 384`.

### 3.1. JEPA

JEPA mới dùng V-JEPA 2.1 ViT-L:

- Ảnh: `384 x 384`.
- Patch size: `16`.
- Token mỗi frame: `24 x 24 = 576`.
- Embed dim: `1024`.
- Cache feature: `.npy`.
- Dtype cache: `float16`.
- Khi train, dataset đọc cache bằng memmap rồi convert sang float để đưa vào model.

Trong `JEPA/src/jepa_wm/engine/encode_patch.py`:

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
    t = enc(x)
toks.append(t.to(torch.float16).cpu().numpy())
```

Ý nghĩa:

- Encoder chạy bằng `bf16` trên GPU.
- Output lưu xuống disk bằng `fp16`.
- Disk cache và RAM đọc feature giảm khoảng một nửa so với `fp32`.

### 3.2. NN-JEPA

NN-JEPA hiện tại đang dùng feature cache:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

Metadata từng được in ra:

```text
encoder_name: vit_base_384
tokens_per_frame: 576
embed_dim: 768
dtype: fp32
```

Thư mục feature hiện tại khoảng:

```text
189G data/processed/features/vjepa2_1_vitb_384_ema_fp32
```

Trong `src/data/feature_sequence_dataset.py`, mỗi frame đọc ra:

```python
frame = np.array(self.feature_array[row], copy=True)
return torch.from_numpy(frame).to(dtype=torch.float32)
```

Ý nghĩa:

- Dù `.npy` là memmap, mỗi frame vẫn bị copy thành numpy array riêng.
- Sau đó luôn convert thành `torch.float32`.
- Một sample nhiều frame sẽ stack nhiều frame `fp32` trong RAM.

---

## 4. Kích Thước Một Sample: Đây Là Khác Biệt Rất Lớn

## 4.0. Giải Thích Từng Tham Số Sample/Feature Của Hai Bên

Hai cụm config đang nói cùng một ý tưởng: một sample train là một đoạn ngắn gồm nhiều frame liên tiếp, mỗi frame đã được encode thành patch tokens bởi frozen V-JEPA encoder. Tuy nhiên tên biến và cách tổ chức của `JEPA` và `NN-JEPA` khác nhau.

### 4.0.1. Cụm config của JEPA

```yaml
horizon: 4
frame_stride: 2
num_tokens: 576
latent_dim: 1024
```

Ý nghĩa từng tham số:

| Tham số JEPA | Ý nghĩa |
|---|---|
| `horizon: 4` | Số frame/latent steps trong một sample train. Một sample có `4` frame được lấy từ cùng một session. |
| `frame_stride: 2` | Khoảng cách lấy frame trong sample. `2` nghĩa là lấy cách 2 dòng/frame một lần: frame `t`, `t+2`, `t+4`, `t+6`. |
| `num_tokens: 576` | Số patch token của mỗi frame sau khi qua V-JEPA encoder. Với ảnh `384x384`, patch size `16`, ta có `24x24=576` token/frame. |
| `latent_dim: 1024` | Chiều vector của mỗi patch token. JEPA của bạn bạn đang dùng V-JEPA 2.1 ViT-L nên mỗi token có `1024` chiều. |

Với config này, một sample JEPA có shape:

```text
tokens: (T, N, D) = (4, 576, 1024)
```

Trong đó:

- `T = horizon = 4`
- `N = num_tokens = 576`
- `D = latent_dim = 1024`

Do `frame_stride=2`, sample không lấy 4 frame sát nhau tuyệt đối, mà lấy 4 frame cách nhau 2 bước. Nếu dữ liệu khoảng 9 FPS, thì stride 2 tạo khoảng cách thời gian xấp xỉ 220 ms giữa hai latent steps. Cách này làm chuyển động giữa hai frame rõ hơn, giúp model học dynamics tốt hơn thay vì chỉ học gần giống identity.

Ví dụ index:

```text
start = 100
horizon = 4
frame_stride = 2

sample frames = [100, 102, 104, 106]
```

### 4.0.2. Vì sao JEPA gọi là `horizon`?

Trong repo `JEPA`, `horizon` được dùng theo nghĩa độ dài đoạn thời gian mà model nhìn/rollout trong latent space. Nó là số latent frames trong một clip train.

Nói đơn giản:

```text
horizon của JEPA ~= raw_frames_per_sample của NN-JEPA
```

Nhưng chỉ nên gọi là **gần tương đương**, không nên gọi là tương đương tuyệt đối.

Lý do:

1. `horizon` trong JEPA mang nghĩa temporal horizon cho dynamics/planning.

   Trong `JEPA`, cùng một ý niệm horizon được dùng rất tự nhiên cho cả train clip, rollout và CEM planning. Nó nói rằng model đang xét một đoạn tương lai dài bao nhiêu bước trong latent space.

   Ví dụ:

   ```text
   horizon = 4
   nghĩa là clip có 4 latent states: z0, z1, z2, z3
   ```

   Khi planning, horizon cũng thường nghĩa là số bước action tương lai cần tối ưu.

2. `raw_frames_per_sample` trong NN-JEPA là tên theo Dataset.

   NN-JEPA đặt tên này để nói rõ một sample được cắt từ bao nhiêu frame raw/manifest trước khi đưa vào model. Nó là tham số dataset trước, còn ý nghĩa rollout/planning là hệ quả phía sau.

   Ví dụ:

   ```text
   raw_frames_per_sample = 8
   nghĩa là Dataset lấy 8 frame để tạo một sample
   ```

3. Hai tham số có thể cho cùng số frame, nhưng số transition/action target khác số frame.

   Nếu có `T` frame:

   ```text
   số latent states = T
   số transition next-frame = T - 1
   số action dùng để nối các frame = T - 1
   ```

   Ví dụ JEPA:

   ```text
   horizon = 4
   frames/states = [z0, z1, z2, z3]
   transitions = z0->z1, z1->z2, z2->z3
   số transition = 3
   ```

   Ví dụ NN-JEPA:

   ```text
   raw_frames_per_sample = 8
   frames/states = [z0, z1, z2, z3, z4, z5, z6, z7]
   transitions = 7
   ```

   Vì vậy nếu gọi đơn giản "horizon = raw_frames_per_sample" thì dễ nhầm rằng action horizon cũng bằng đúng số frame. Thực tế action/transition horizon thường là `T - 1`.

4. JEPA có `frame_stride`, còn NN-JEPA hiện có `sequence_stride`, nhưng hai tham số này **không giống nhau tuyệt đối trong code hiện tại**.

   Trong `JEPA`, `frame_stride` là khoảng cách giữa các frame bên trong cùng một sample:

   ```text
   JEPA: horizon=4, frame_stride=2
   sample = [100, 102, 104, 106]
   ```

   Trong `NN-JEPA` hiện tại, `sequence_stride` trong `build_sequence_windows(...)` là bước trượt cửa sổ start index, không phải khoảng cách giữa các frame bên trong sample. Với `raw_frames_per_sample=4`, `sequence_stride=2`, các sample sẽ giống:

   ```text
   sample 1 = [100, 101, 102, 103]
   sample 2 = [102, 103, 104, 105]
   sample 3 = [104, 105, 106, 107]
   ```

   Tức là `sequence_stride` của NN-JEPA hiện tại làm giảm/tăng số lượng sample overlap, chứ không tạo sample kiểu `[100, 102, 104, 106]`.

   Vì vậy muốn NN-JEPA thật sự giống `frame_stride=2` của JEPA thì cần thêm một tham số mới, ví dụ:

   ```text
   frame_stride_inside_sample: 2
   ```

   hoặc sửa `build_sequence_windows(...)` để window lấy frame cách nhau `sample_frame_stride`.

   Nếu có frame stride bên trong sample, độ dài thời gian thật xấp xỉ:

   ```text
   temporal span = (T - 1) * frame_stride_inside_sample / FPS
   ```

   Hiện NN-JEPA đang dùng frame liên tiếp bên trong sample, nên frame stride bên trong sample thực tế là `1`.

5. Loss implementation cũng không dùng toàn bộ sample theo cùng cách.

   Trong JEPA, train loss gồm:

   ```text
   teacher forcing: dự đoán z1,z2,z3 từ z0,z1,z2
   rollout: chủ yếu 2-step rollout z0->z1->z2
   ```

   Trong NN-JEPA:

   ```text
   teacher forcing: dự đoán toàn bộ z1..z7 từ z0..z6 nếu T=8
   rollout: dùng auto_steps, hiện thường = 2
   ```

   Tức là `raw_frames_per_sample=8` làm teacher-forcing sequence dài hơn, nhưng rollout loss hiện vẫn chỉ dùng `auto_steps=2`. Vì vậy sample có 8 frame không có nghĩa rollout đang rollout đủ 7 bước.

6. Trong planning/inference, chữ horizon thường có nghĩa khác với số frame train.

   Khi nói CEM horizon, người ta thường hiểu:

   ```text
   số action tương lai cần tối ưu
   ```

   Còn `raw_frames_per_sample` chỉ là cách dataset cắt clip khi train. Vì vậy nếu dùng chữ `horizon` cho NN-JEPA, cần nói rõ là:

   ```text
   train context/sample horizon
   ```

   không phải nhất thiết là:

   ```text
   planning horizon
   ```

Trong bài toán này, hai tên đó gần như cùng vai trò: số frame/latent steps trong một sample train.

Nói chính xác hơn:

```text
JEPA horizon tương đương gần nhất với NN-JEPA raw_frames_per_sample ở mức "số latent frames trong một training sample".
```

Nhưng chúng không hoàn toàn tương đương vì:

- tên gọi nhấn mạnh mục đích khác nhau
- stride khác nhau làm độ dài thời gian thật khác nhau
- số transition/action là `T-1`, không phải `T`
- rollout loss có thể chỉ dùng `auto_steps`, không dùng hết toàn bộ frame
- planning horizon không nhất thiết bằng train sample length

### 4.0.3. Cụm config của NN-JEPA

```yaml
raw_frames_per_sample: 8
sequence_stride: 1
tokens_per_frame: 576
embed_dim: 768
feature dtype: fp32
```

Ý nghĩa từng tham số:

| Tham số NN-JEPA | Ý nghĩa |
|---|---|
| `raw_frames_per_sample: 8` | Số frame trong một sample train. Một sample có `8` frame liên tiếp từ cùng session. |
| `sequence_stride: 1` | Bước trượt cửa sổ khi tạo dataset windows. Trong code hiện tại, nó không phải khoảng cách giữa các frame trong cùng sample. Với `1`, các window bắt đầu ở mọi frame hợp lệ. |
| `tokens_per_frame: 576` | Số patch token của mỗi frame sau V-JEPA encoder. Vì NN-JEPA hiện cũng dùng ảnh `384x384`, patch size `16`, nên cũng là `576` token/frame. |
| `embed_dim: 768` | Chiều vector của mỗi patch token. NN-JEPA hiện dùng V-JEPA 2.1 ViT-B, nên mỗi token có `768` chiều. |
| `feature dtype: fp32` | Kiểu số của feature cache đã lưu trên disk và đưa vào train. `fp32` nghĩa là float32, 4 bytes/số. |

Với config này, một sample NN-JEPA có shape:

```text
latents: (T * N, D) = (8 * 576, 768) = (4608, 768)
```

Nếu reshape về dạng dễ hiểu:

```text
(T, N, D) = (8, 576, 768)
```

Trong đó:

- `T = raw_frames_per_sample = 8`
- `N = tokens_per_frame = 576`
- `D = embed_dim = 768`

Trong code hiện tại, sample lấy 8 frame liền nhau vì `build_sequence_windows(...)` lấy lát cắt liên tiếp:

```python
window = indices[start : start + raw_frames_per_sample]
```

`sequence_stride=1` chỉ làm các start index trượt từng frame một.

Ví dụ index:

```text
start = 100
raw_frames_per_sample = 8
sequence_stride = 1

sample frames = [100, 101, 102, 103, 104, 105, 106, 107]
```

Nếu `sequence_stride=2`, code hiện tại sẽ không tạo sample `[100, 102, 104, ...]`. Nó sẽ tạo các sample overlap ít hơn:

```text
sample 1 = [100, 101, 102, 103, 104, 105, 106, 107]
sample 2 = [102, 103, 104, 105, 106, 107, 108, 109]
sample 3 = [104, 105, 106, 107, 108, 109, 110, 111]
```

### 4.0.4. Bảng đối chiếu tên tham số giữa hai repo

| JEPA | NN-JEPA | Có tương đương không? | Ghi chú |
|---|---|---|---|
| `horizon` | `raw_frames_per_sample` | Gần tương đương | Cả hai đều là số frame/latent steps trong một sample. |
| `frame_stride` | Chưa có tham số tương đương trực tiếp | Không tương đương | JEPA dùng để lấy frame cách nhau bên trong sample. NN-JEPA hiện luôn lấy frame liên tiếp bên trong sample. |
| Bước trượt cửa sổ dataset | `sequence_stride` | Tương đương với stride của start index | NN-JEPA dùng để quyết định sample tiếp theo bắt đầu cách sample trước bao nhiêu frame. |
| `num_tokens` | `tokens_per_frame` | Tương đương | Cả hai đều là số patch token mỗi frame. |
| `latent_dim` | `embed_dim` | Tương đương về ý nghĩa | Cả hai đều là chiều vector mỗi token, nhưng khác giá trị do encoder khác nhau. |
| cache dtype `fp16` | feature dtype `fp32` | Không giống | JEPA lưu feature nhẹ hơn, NN-JEPA hiện lưu chính xác hơn nhưng nặng hơn. |

### 4.0.5. Khác biệt thực tế của hai sample

JEPA:

```text
(4, 576, 1024), stride 2, fp16 cache
```

NN-JEPA:

```text
(8, 576, 768), stride 1, fp32 cache
```

So sánh số phần tử latent mỗi sample:

```text
JEPA elements = 4 * 576 * 1024 = 2,359,296
NN-JEPA elements = 8 * 576 * 768 = 3,538,944
```

NN-JEPA có nhiều phần tử hơn:

```text
3,538,944 / 2,359,296 = 1.5 lần
```

Nhưng nếu tính số bytes theo dtype cache:

```text
JEPA fp16 bytes = 2,359,296 * 2 ≈ 4.5 MB/sample
NN-JEPA fp32 bytes = 3,538,944 * 4 ≈ 13.5 MB/sample
```

NN-JEPA nặng hơn về dữ liệu input cache khoảng:

```text
13.5 / 4.5 = 3 lần
```

Đây là một trong những lý do DataLoader của NN-JEPA dễ làm đầy RAM hơn.

### 4.0.6. Khác biệt attention còn lớn hơn khác biệt input bytes

Predictor không chỉ đọc latent input. Nó còn chạy transformer attention trên chuỗi token.

Vì mỗi frame có thêm action token và state token:

```text
tokens_per_step = 576 + 2 = 578
```

JEPA:

```text
sequence length L = horizon * tokens_per_step
L = 4 * 578 = 2312
```

NN-JEPA:

```text
sequence length L = raw_frames_per_sample * tokens_per_step
L = 8 * 578 = 4624
```

Attention có chi phí gần với `L^2`, nên:

```text
(4624 / 2312)^2 = 4
```

Tức là chỉ vì NN-JEPA dùng `8` frame thay vì `4` frame, phần attention có thể nặng khoảng `4 lần` cho một forward teacher-forcing cùng batch size, dù NN-JEPA dùng `embed_dim=768` thấp hơn `1024`.

### 4.0.7. Vậy bên nào "mạnh" hơn?

Không thể kết luận chỉ từ các con số này.

JEPA có:

- encoder ViT-L mạnh hơn, `latent_dim=1024`
- predictor lớn hơn
- feature cache nhẹ hơn vì `fp16`
- sample ngắn hơn, stride xa hơn

NN-JEPA có:

- encoder ViT-B nhẹ hơn, `embed_dim=768`
- sample dài hơn `8` frame
- feature cache `fp32` chính xác hơn nhưng nặng hơn
- frame bên trong sample hiện đang sát nhau hơn vì NN-JEPA chưa có tham số frame-stride bên trong sample

Về chất lượng dynamics, `frame_stride=2` có thể có lợi vì frame cách nhau đủ xa để thấy chuyển động. Nếu frame quá sát nhau, model dễ học kiểu "frame sau gần giống frame trước", val loss có thể đẹp nhưng planning/control chưa chắc tốt.

Về memory, JEPA rõ ràng tối ưu hơn.

### 4.1. JEPA

JEPA config:

```yaml
horizon: 4
frame_stride: 2
num_tokens: 576
latent_dim: 1024
```

Một sample JEPA có shape:

```text
T x N x D = 4 x 576 x 1024
```

Nếu cache fp16:

```text
4 * 576 * 1024 * 2 bytes ≈ 4.5 MB/sample
```

Khi vào model, do train dùng `bf16 autocast`, activation cũng nhẹ hơn đáng kể.

### 4.2. NN-JEPA

NN-JEPA tiny newdata config:

```yaml
raw_frames_per_sample: 8
sequence_stride: 1
tokens_per_frame: 576
embed_dim: 768
feature dtype: fp32
```

Một sample NN-JEPA có shape:

```text
T x N x D = 8 x 576 x 768
```

Với `fp32`:

```text
8 * 576 * 768 * 4 bytes ≈ 13.5 MB/sample
```

Với `batch_size=32`:

```text
13.5 MB * 32 ≈ 432 MB
```

Đây mới chỉ là tensor latent đầu vào. Chưa tính:

- target latent
- activation trong transformer
- attention matrix
- rollout forward nhiều lần
- gradient
- optimizer state
- DataLoader prefetch
- pinned memory
- W&B gradient/parameter logging

---

## 5. Attention Memory: Vì Sao 8 Frame Nặng Hơn Nhiều So Với 4 Frame?

Transformer attention thường có chi phí theo bình phương sequence length.

### 5.1. JEPA

JEPA dùng mỗi frame:

```text
576 patch tokens + 2 condition tokens = 578 tokens/frame
```

Với `horizon=4`:

```text
L = 4 * 578 = 2312 tokens
```

### 5.2. NN-JEPA

NN-JEPA hiện tại cũng có:

```text
576 patch tokens + 2 condition tokens = 578 tokens/frame
```

Nhưng với `raw_frames_per_sample=8`:

```text
L = 8 * 578 = 4624 tokens
```

So sánh attention:

```text
(4624 / 2312)^2 = 4x
```

Tức là NN-JEPA có sequence dài gấp đôi nhưng attention memory/work có thể nặng khoảng gấp bốn lần cho phần teacher forcing.

Đây là lý do rất quan trọng: số tham số model nhỏ không đảm bảo train nhẹ nếu sequence length quá dài.

---

## 6. Train/Eval Dtype: JEPA Dùng bf16, NN-JEPA Đang fp32

### 6.1. JEPA

Trong `JEPA/src/jepa_wm/engine/train_ac_car.py`, cả train và val đều dùng:

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    loss, tf, ro = _losses(model, b, device)
```

Điều này giúp:

- giảm VRAM activation
- tăng tốc Tensor Core trên GPU hỗ trợ bf16
- giảm khả năng OOM lúc val
- vẫn ổn định hơn fp16 vì bf16 giữ exponent rộng hơn

### 6.2. NN-JEPA

Trong `src/tools/train_rc_jepa_ac_features.py`, hiện tại không có `torch.autocast`.

Tức là:

- model forward fp32
- loss fp32
- activation fp32
- validation cũng fp32

Vì vậy dù predictor NN-JEPA tiny nhỏ, activation và attention buffer vẫn lớn.

---

## 7. DataLoader: Khác Biệt Lớn Nhất Về RAM

### 7.1. JEPA Dùng SessionBatchSampler

Trong `JEPA/src/jepa_wm/data/ac_clip.py`, JEPA có:

```python
class SessionBatchSampler(Sampler):
    """Yield batches whose items all come from ONE session..."""
```

Ý tưởng:

- Mỗi batch chỉ lấy sample từ một session.
- Session order shuffle theo epoch.
- Window trong cùng session cũng shuffle.
- Cache memmap theo session được giữ nóng.
- Tránh trường hợp batch 1 đọc session A, batch 2 đọc session Z, batch 3 đọc session B liên tục.

Kèm theo:

```python
@lru_cache(maxsize=2)
def _load_tokens(npy_path):
    return np.load(npy_path, mmap_mode="r")
```

Tức là chỉ giữ cache nhỏ khoảng 2 session memmap gần nhất.

### 7.2. NN-JEPA Shuffle Toàn Cục

Trong `src/data/feature_sequence_dataset.py`, NN-JEPA:

```python
self.session_features = {
    session_id: load_session_feature_index(...)
    for session_id in sorted(session_ids)
}
```

Và DataLoader:

```python
"train": DataLoader(..., batch_size=batch_size, shuffle=settings.SHUFFLE_TRAIN, ...)
```

Ý nghĩa:

- Dataset mở memmap cho tất cả session trong split.
- Train shuffle toàn cục theo sample.
- Một batch có thể chứa nhiều session khác nhau.
- Nhiều worker có thể nhảy giữa nhiều file `.npy`.
- Prefetch sẽ copy nhiều sample fp32 vào RAM trước.

Với feature cache lớn, cách này dễ tạo áp lực RAM và I/O hơn JEPA.

---

## 8. DataLoader Prefetch, Pin Memory, Persistent Workers

NN-JEPA settings:

```python
BATCH_SIZE = 32
AC_EVAL_BATCH_SIZE = 2
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
SHUFFLE_TRAIN = True
```

Trong Hydra `rc_jepa_tiny_newdata.yaml`, user đang override:

```yaml
batch_size: 32
eval_batch_size: 2
num_workers: 8
```

Với PyTorch DataLoader, khi `num_workers=8` và `prefetch_factor=4`, mỗi worker có thể chuẩn bị trước nhiều batch. Tổng lượng dữ liệu nằm trong RAM có thể cao hơn nhiều so với một batch thật đang train.

Ước lượng rất thô:

```text
1 sample NN-JEPA ≈ 13.5 MB latent fp32
batch 32 ≈ 432 MB latent
8 workers * prefetch_factor 4 = tối đa 32 batch đang/đã chuẩn bị
432 MB * 32 ≈ 13.8 GB chỉ riêng latent copy lý thuyết
```

Con số thực tế có thể thấp hơn vì DataLoader scheduling, nhưng nó giải thích vì sao Linux OOM killer từng giết `pt_data_worker` và `python3`.

JEPA thì khác:

```python
train_dl = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=nw, pin_memory=True)
val_dl = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=nw)
```

Điểm khác:

- Không set `persistent_workers=True`.
- Không set `prefetch_factor=4` thủ công.
- Val không bật `pin_memory=True`.
- Batch theo session nên ít nhảy file.

---

## 9. Vì Sao Train Batch Lớn Có Thể Chạy Nhưng Val Vẫn OOM?

Có hai loại OOM cần phân biệt:

### 9.1. CUDA OOM

CUDA OOM thường có traceback dạng:

```text
torch.OutOfMemoryError: CUDA out of memory
```

Lỗi này xảy ra khi VRAM GPU không đủ.

### 9.2. System RAM OOM

System RAM OOM thường không có traceback Python rõ ràng. Linux kernel sẽ giết process:

```text
Out of memory: Killed process ... (python3)
Out of memory: Killed process ... (pt_data_worker)
```

Với hai run gần nhất của NN-JEPA, log kernel cho thấy bị system RAM OOM, không phải CUDA OOM.

Val batch size nhỏ hơn train nhưng vẫn có thể OOM vì:

- Worker/prefetch từ train có thể còn sống nếu `persistent_workers=True`.
- DataLoader val cũng dùng `num_workers`, `pin_memory`, `prefetch_factor`.
- W&B watch/grad stats có thể giữ thêm tensor/metadata.
- PyTorch CUDA cache chưa chắc trả hết ngay sau train.
- Validation forward fp32 vẫn tạo attention buffer lớn.
- Nếu eval vừa bắt đầu sau train, RAM đã ở trạng thái cao.

Vì vậy giảm `eval_batch_size` từ 16 xuống 2 chỉ xử lý một phần. Nếu RAM OOM đến từ DataLoader worker, cần giảm `num_workers`, `prefetch_factor`, tắt `persistent_workers`, hoặc chuyển sang session sampler.

---

## 10. W&B Logging Cũng Là Một Khác Biệt

JEPA train loop log nhẹ:

```python
if run and gstep % 50 == 0:
    run.log({"train/loss": ..., "train/tf": ..., "train/rollout": ..., "train/lr": ...})
```

NN-JEPA hiện tại có:

```yaml
watch_log: gradients
watch_freq: 200
grad_stats_every: 20
param_stats_every: 200
```

Và trong train loop có collect gradient metrics, parameter metrics.

Điều này không phải nguyên nhân chính, nhưng làm tăng overhead:

- thêm CPU time
- thêm RAM
- thêm W&B background process
- thêm khả năng system RAM bị áp lực khi DataLoader đã nặng

Khi train overnight, nên giảm logging trước để ổn định:

```yaml
watch_log: none
grad_stats_every: 0
param_stats_every: 0
```

Sau khi pipeline ổn mới bật lại logging sâu.

---

## 11. Loss Và Train Objective Có Giống Nhau Không?

### 11.1. JEPA

Trong `JEPA/src/jepa_wm/engine/train_ac_car.py`, loss:

- teacher forcing L1
- 2-step rollout L1

Code:

```python
out = model(z, a, s)
tf = F.l1_loss(out[:, :-1], z[:, 1:])

p1 = model(z[:, :1], a[:, :1], s[:, :1])[:, -1:]
ctx = torch.cat([z[:, :1], p1], dim=1)
p2 = model(ctx, a[:, :2], s[:, :2])[:, -1:]
ro = F.l1_loss(p2[:, 0], z[:, 2])
```

### 11.2. NN-JEPA

Trong `src/models/rc_jepa_ac.py`, loss cũng là:

- teacher forcing L1
- rollout L1

Nhưng NN-JEPA rollout tổng quát theo `auto_steps`:

```python
rollout_steps = min(auto_steps, num_frames - 1)
```

Với `auto_steps=2`, về tinh thần giống JEPA.

Khác biệt chính không nằm ở loss, mà nằm ở:

- số frame input
- dtype
- DataLoader
- feature cache
- architecture chi tiết

---

## 12. Kiến Trúc Predictor: Giống Và Khác

### 12.1. JEPA VJEPA2ACCar

`JEPA/src/jepa_wm/models/vjepa2_ac_car.py`:

- input: patch token map `(B, T, N, D)`
- action token
- state token
- block-causal transformer
- output patch token prediction
- optional residual
- rollout autoregressive

Mỗi frame có group:

```text
[action_t, state_t, patch_t_1, ..., patch_t_N]
```

Mask:

```text
frame t chỉ attend được frame <= t
```

### 12.2. NN-JEPA Simple

NN-JEPA `SimpleACPredictor` cũng dùng:

- action token
- state token
- patch token
- time-causal mask
- transformer encoder
- output patch token prediction

Nhưng implementation đơn giản hơn:

- dùng `nn.TransformerEncoder`
- không có RoPE style attention
- không có session-specific sampler
- không có bf16 autocast trong trainer hiện tại

### 12.3. NN-JEPA Official-Lite

NN-JEPA đã có `VJepaStyleACPredictor`:

- gần Meta hơn `SimpleACPredictor`
- có action-block causal attention mask
- có attention riêng hơn thay vì chỉ dùng `nn.TransformerEncoder`
- vẫn là bản local simplified, không y chang Meta tuyệt đối

Nhưng nếu train `official_lite` với pipeline hiện tại mà không sửa dtype/DataLoader/sample length thì vẫn có thể OOM. Architecture không tự giải quyết vấn đề DataLoader RAM.

---

## 13. Vì Sao JEPA 26M/39M Có Thể Ổn Hơn NN-JEPA Tiny 0.67M?

Lý do chính:

### 13.1. Param memory nhỏ hơn activation memory trong bài này

Với tiny NN-JEPA:

- tham số chỉ `0.67M`
- nhưng input latent batch rất lớn
- sequence length 4624
- attention quadratic
- DataLoader prefetch nhiều batch fp32

Với JEPA:

- tham số có thể `26M` hoặc `39M`
- nhưng sample ngắn hơn
- bf16 activation
- fp16 cache
- session batching
- ít prefetch hơn

Do đó tổng RAM/VRAM có thể thấp hơn.

### 13.2. Dataset pipeline quan trọng hơn model size

Trong bài toán patch-token JEPA-AC, memory dominated by:

```text
batch_size * frames * tokens_per_frame * dim * dtype
```

và:

```text
batch_size * heads * sequence_length^2
```

Không chỉ bởi:

```text
number_of_model_parameters
```

---

## 14. Những Điểm JEPA Làm Tốt Nên Port Sang NN-JEPA

### 14.0. Dataset Được Hình Thành Như Thế Nào?

Phần này giải thích kỹ câu:

```text
Ưu tiên 1: thêm session-batch sampler cho feature dataset
```

Cần phân biệt ba khái niệm:

| Khái niệm | Ý nghĩa |
|---|---|
| session | Một lần chạy/ghi dữ liệu của xe, ví dụ `session_20260607_...`, gồm frames + actions + imu/gps. |
| sample/window/clip | Một đoạn ngắn cắt từ một session, ví dụ 8 frame liên tiếp hoặc 4 frame cách nhau stride. Đây là một item của Dataset. |
| batch | Nhiều sample gom lại để train một bước optimizer. |

Điểm quan trọng: cả NN-JEPA hiện tại và JEPA của bạn bạn đều đúng ở mức **mỗi sample phải nằm trong một session**, không được ghép frame từ nhiều session khác nhau trong một sample. Khác biệt nằm ở **batch được gom như thế nào**.

---

### 14.0.1. NN-JEPA hiện tại hình thành feature dataset như thế nào?

Pipeline NN-JEPA hiện tại:

```text
data/raw/session_x
  frames/*.jpg
  actions_synced.csv
  imu_synced.csv
  gps.csv
        |
        v
preprocess
        |
        v
data/processed/manifests/train.jsonl
data/processed/manifests/val.jsonl
data/processed/manifests/test.jsonl
        |
        v
extract_vjepa_features
        |
        v
data/processed/features/.../sessions/session_x.npy
data/processed/features/.../sessions/session_x.json
        |
        v
RCJepaACFeatureSequenceDataset
        |
        v
DataLoader -> batch -> train loop
```

Manifest `.jsonl` là danh sách frame đã xử lý. Mỗi dòng thường chứa:

- `session_id`
- `frame_index`
- `frame_path`
- `state`
- `action`
- timestamp

Feature extractor tạo cache theo session:

```text
session_x.npy  = feature array shape [N_frames, tokens_per_frame, embed_dim]
session_x.json = mapping frame_index -> row trong .npy
```

Dataset hiện tại đọc manifest rồi tạo windows bằng `build_sequence_windows(...)`.

Với config hiện tại:

```yaml
raw_frames_per_sample: 8
sequence_stride: 1
```

Nó tạo sample kiểu:

```text
sample 1 = [frame 100, 101, 102, 103, 104, 105, 106, 107]
sample 2 = [frame 101, 102, 103, 104, 105, 106, 107, 108]
sample 3 = [frame 102, 103, 104, 105, 106, 107, 108, 109]
```

Nếu `sequence_stride=2`, hiện tại nó sẽ tạo:

```text
sample 1 = [100, 101, 102, 103, 104, 105, 106, 107]
sample 2 = [102, 103, 104, 105, 106, 107, 108, 109]
sample 3 = [104, 105, 106, 107, 108, 109, 110, 111]
```

Nghĩa là `sequence_stride` hiện tại là stride của **start index**, không phải stride giữa frame bên trong sample.

Sau đó `DataLoader` hiện tại dùng:

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
```

Vì `shuffle=True` toàn cục, một batch có thể chứa sample từ nhiều session:

```text
batch = [
  sample từ session_A,
  sample từ session_K,
  sample từ session_B,
  sample từ session_A,
  sample từ session_Z,
  ...
]
```

Về mặt học máy, cách này không sai. Random minibatch toàn cục là cách rất phổ biến.

Nhưng với feature cache lớn theo session `.npy`, cách này có nhược điểm:

- worker phải nhảy qua nhiều file `.npy`
- cache memmap khó giữ nóng
- prefetch có thể copy nhiều batch lớn vào RAM
- pinned memory + persistent workers làm RAM dễ tăng
- batch có thể cần đọc nhiều session khác nhau cùng lúc

Đây là lý do NN-JEPA hiện tại dễ bị system RAM OOM dù model tiny ít tham số.

---

### 14.0.2. JEPA của bạn bạn hình thành dataset như thế nào?

Repo `JEPA` của bạn bạn dùng `ACClipDataset`.

Pipeline:

```text
raw_towerpro/session_x
  frames/*.jpg
  actions_synced.csv
  imu_synced.csv
  gps.csv
        |
        v
encode_patch.py
        |
        v
data/latents_towerpro_patch_384/session_x.npy
        |
        v
ACClipDataset
        |
        v
SessionBatchSampler
        |
        v
DataLoader -> batch -> train loop
```

Với config:

```yaml
horizon: 4
frame_stride: 2
```

Dataset tạo sample kiểu:

```text
sample = [frame 100, 102, 104, 106]
```

Điểm khác quan trọng: JEPA có `SessionBatchSampler`.

Sampler này group index theo session:

```text
session_A -> [sample indices của session_A]
session_B -> [sample indices của session_B]
session_C -> [sample indices của session_C]
```

Khi tạo batch, nó tạo batch trong cùng một session:

```text
batch 1 = [sample_A_01, sample_A_07, sample_A_23, ...]
batch 2 = [sample_A_11, sample_A_05, sample_A_42, ...]
batch 3 = [sample_K_02, sample_K_19, sample_K_31, ...]
```

Nó vẫn shuffle:

- shuffle thứ tự session mỗi epoch
- shuffle sample/window bên trong session

Nhưng nó tránh batch kiểu nhảy loạn nhiều session:

```text
Không ưu tiên: [session_A, session_K, session_Z, session_B, ...] trong cùng batch
Ưu tiên:      [session_A, session_A, session_A, session_A, ...] trong cùng batch
```

Lợi ích:

- đọc tập trung vào một file `.npy`
- memmap cache nóng hơn
- giảm random I/O
- giảm áp lực RAM/prefetch
- dễ ổn định hơn khi feature cache rất lớn

Tradeoff:

- batch ít đa dạng session hơn tại từng optimizer step
- gradient mỗi step có thể hơi correlated theo session
- cần shuffle session và shuffle window trong session để tránh bias
- nếu số session ít, batch theo session có thể làm train kém đa dạng hơn

Với dữ liệu xe RC có nhiều session, tradeoff này thường chấp nhận được, vì lợi ích ổn định RAM/I/O rất lớn.

---

### 14.0.3. Source public V-JEPA2 AC của Meta làm dataset như thế nào?

Trong source public `vjepa2/app/vjepa_droid`, Meta không train từ feature cache `.npy` như NN-JEPA/JEPA của mình. Họ đọc raw video/trajectory rồi encode target bằng encoder trong train loop.

Dataset chính:

```text
vjepa2/app/vjepa_droid/droid.py
```

Ý tưởng:

1. CSV dataset chứa danh sách trajectory directory.
2. `DROIDVideoDataset.__getitem__(index)` chọn một trajectory/session.
3. Nó đọc video bằng Decord.
4. Nó sample một random window trong video.
5. Nó lấy frames, states, extrinsics.
6. Nó suy ra actions từ pose differences.
7. Trả về:

```text
buffer, actions, states, extrinsics, indices
```

Shape trong train loop:

```python
clips   # [B, C, T, H, W]
actions # [B, T-1, 7]
states  # [B, T, 7]
extrinsics # [B, T, 7]
```

DataLoader official dùng:

```python
DistributedSampler(dataset, shuffle=True)
DataLoader(..., sampler=dist_sampler, batch_size=batch_size, ...)
```

Tức là Meta shuffle ở mức trajectory/sample bằng distributed sampler. Họ không có `SessionBatchSampler` kiểu batch chỉ cùng session.

Tuy nhiên cần hiểu bối cảnh:

- Meta đọc raw video trực tiếp.
- Mỗi item đã là một clip random từ một trajectory.
- Họ chạy trên multi-node/multi-GPU, config public còn ghi `mem_per_gpu: 220G`.
- Họ không phải đọc feature cache `.npy` khổng lồ từ disk theo kiểu memmap.

Vì vậy việc họ không dùng session-batch sampler không có nghĩa session-batch sampler là sai. Nó chỉ không cần thiết trong pipeline raw-video/distributed của họ.

---

### 14.0.4. Ai đúng hơn: NN-JEPA hiện tại hay JEPA của bạn bạn?

Câu trả lời phải tách thành hai tiêu chí.

#### Theo mặt thuật toán/paper

Cả hai đều có thể đúng nếu đảm bảo:

- sample là clip liên tục/có thứ tự từ cùng trajectory/session
- action nối đúng transition giữa frame
- state/action đồng bộ đúng thời điểm
- train/val/test split theo session để tránh leakage
- predictor nhận latent tokens + action/state và học next latent

Ở tiêu chí này, NN-JEPA hiện tại không sai. Nó tạo sliding windows từ manifest và shuffle minibatch toàn cục. Đây là cách rất chuẩn trong machine learning.

JEPA của bạn bạn cũng không sai. Nó cũng tạo window từ session, chỉ khác là batch được gom theo session để tối ưu I/O.

#### Theo mặt engineering cho feature-cache lớn

JEPA của bạn bạn hợp lý hơn.

Lý do:

- feature cache lưu theo session `.npy`
- mỗi session có thể rất lớn
- memmap hoạt động tốt nhất khi đọc tương đối tập trung
- batch cùng session giảm nhảy file
- giảm RAM/I/O pressure
- ổn định hơn khi train overnight

NN-JEPA hiện tại dùng random shuffle toàn cục, đúng về thống kê nhưng kém tối ưu cho memmap feature cache lớn.

---

### 14.0.5. Cái nào gần source V-JEPA2 AC hơn?

Không cái nào y chang source Meta, vì cả NN-JEPA và JEPA của bạn bạn đều train từ precomputed feature cache, còn Meta public train từ raw video và encode trong train loop.

Đọc trực tiếp source public `vjepa2/app/vjepa_droid`:

- Config public `vjepa2/configs/train/vitg16/droid-256px-8f.yaml` dùng `dataset_fpcs: [8]`, `fps: 4`, `crop_size: 256`, `patch_size: 16`, `batch_size: 8`, `dtype: bfloat16`.
- Trong `train.py`, `max_num_frames = max(dataset_fpcs)`, nên clip train có `T=8`.
- Trong `train.py`, DataLoader được tạo bằng `init_data(... frames_per_clip=max_num_frames, tubelet_size=1, fps=fps, ...)`.
- Trong `droid.py`, dataset đọc danh sách trajectory từ CSV, mỗi `__getitem__` chọn một trajectory/session rồi random một window trong video.
- Frame indices được lấy bằng `indices = np.arange(sf, sf + nframes, fstp)`, với `fstp = ceil(vfps / fps)`. Nghĩa là source Meta không lấy mọi frame sát nhau theo raw FPS, mà lấy frame theo target FPS, ví dụ config là `4 FPS`.
- `DistributedSampler(... shuffle=True)` được dùng cho DataLoader. Không thấy session-batch sampler kiểu "một batch chỉ cùng một session".
- Trong train loop, `clips` có shape `[B, C, T, H, W]`, `actions` là `[B, T-1, 7]`, `states` là `[B, T, 7]`.
- Target latent được encode bằng `target_encoder` ngay trong train loop, không phải đọc precomputed feature `.npy`.

Một chi tiết dễ nhầm: config có `tubelet_size: 2`, nhưng khi tạo data loader trong `train.py`, source truyền `tubelet_size=1` vào `init_data`. Sau đó trong `forward_target`, mỗi frame được `unsqueeze(2).repeat(..., 2, ...)` để tạo pseudo-clip 2 frame cho encoder tubelet. Vì vậy với V-JEPA2 AC public này, temporal clip của dataset vẫn là `T=8` frame sampled theo `fps=4`; tubelet size không có nghĩa là sample chỉ còn 4 bước train.

So gần đúng:

| Tiêu chí | Meta V-JEPA2 AC public | JEPA của bạn bạn | NN-JEPA hiện tại |
|---|---|---|---|
| Input train | raw video clip | precomputed patch tokens | precomputed patch tokens |
| Mỗi sample là clip từ một trajectory/session | Có | Có | Có |
| Random window trong trajectory | Có | Gần giống, tạo nhiều windows | Có, tạo sliding windows |
| Batch chỉ cùng session | Không thấy | Có | Không |
| Feature cache memmap | Không | Có | Có |
| Action/state theo từng frame | Có | Có | Có |
| Số frame train/sample | `T=8` | `horizon=4` hiện tại | `raw_frames_per_sample=8` hiện tại |
| Temporal spacing trong clip | Có qua `fps=4` | Có qua `frame_stride=2` | Hiện lấy frame liên tiếp trong manifest |
| Batch sampler | `DistributedSampler(shuffle=True)` | `SessionBatchSampler` | DataLoader `shuffle=True` |
| Mixed precision | Có `bfloat16` | Có `bfloat16` | Hiện train feature-cache chủ yếu fp32 |

Nếu xét riêng “đúng với source Meta”:

- Meta không dùng session-batch sampler.
- Meta dùng DistributedSampler shuffle dataset.
- Meta sample clip từ trajectory/video.
- Meta dùng `T=8` frame/sample trong config public DROID.
- Meta dùng target FPS (`fps=4`) để frame cách nhau đủ xa theo thời gian.

Nếu xét “đúng với bài toán feature cache trên máy local”:

- JEPA của bạn bạn thực dụng và ổn định hơn.
- Session-batch sampler là cải tiến engineering hợp lý.
- Nó không làm sai sample/loss, chỉ thay đổi cách gom sample vào batch.

Kết luận chính xác:

```text
NN-JEPA hiện tại đúng về thuật toán nhưng chưa tối ưu cho feature-cache lớn.
JEPA của bạn bạn không y chang source Meta, nhưng đúng hơn về engineering cho cache .npy lớn và ổn định RAM/I/O.
```

Kết luận chặt hơn nếu bắt buộc chọn "bên nào chuẩn source hơn":

```text
Về batch sampler: NN-JEPA hiện tại gần source Meta hơn, vì source Meta shuffle sample toàn cục bằng DistributedSampler; không batch cùng session.
```

```text
Về số frame mỗi sample: NN-JEPA hiện tại gần source Meta hơn, vì cả hai đang dùng 8 frame/sample.
```

```text
Về temporal spacing/fps giữa các frame: JEPA của bạn bạn gần tinh thần source hơn, vì source Meta lấy clip theo target fps=4; NN-JEPA hiện lấy frame liên tiếp trong manifest, chưa có frame_stride_inside_sample.
```

```text
Về train memory engineering trên feature cache: JEPA của bạn bạn tốt hơn, nhưng đây là tối ưu local, không phải behavior y chang source Meta.
```

Do đó không nên nói "JEPA của bạn bạn chuẩn source hơn hoàn toàn" hoặc "NN-JEPA chuẩn source hơn hoàn toàn". Câu đúng là:

```text
NN-JEPA hiện tại gần source hơn ở sampler và số frame T=8.
JEPA của bạn bạn gần source hơn ở ý tưởng temporal downsampling/fps và mixed precision, đồng thời tốt hơn cho feature-cache engineering.
```

---

### 14.0.5.1. Giải thích kỹ Temporal Spacing / FPS

`Temporal spacing` nghĩa là khoảng cách thời gian thật giữa hai frame liên tiếp trong một sample train.

Nó không chỉ là "sample có bao nhiêu frame", mà còn là:

```text
hai frame cạnh nhau trong sample cách nhau bao nhiêu mili-giây?
```

Trong world model/action-conditioned dynamics, điểm này rất quan trọng. Nếu hai frame quá sát nhau, ảnh gần như không đổi:

```text
z_{t+1} gần giống z_t
```

Khi đó predictor có thể học gần như copy latent hiện tại sang latent tiếp theo. Loss có thể giảm nhưng model chưa chắc học tốt tác động của action. Với xe tự lái RC, điều ta cần không chỉ là "frame sau giống frame trước", mà là:

```text
action_t + state_t làm cảnh nhìn thấy ở t+1 thay đổi như thế nào?
```

Nếu frame cách nhau đủ xa, chuyển động do throttle/steering tạo ra rõ hơn trong latent. Model khó hơn, nhưng học dynamics có ý nghĩa hơn.

---

#### Source V-JEPA2 AC public làm gì?

Trong `vjepa2/app/vjepa_droid/droid.py`, source không lấy mọi frame sát nhau từ video. Nó lấy frame theo target FPS.

Logic chính:

```python
vfps = vr.get_avg_fps()
fps = self.fps if self.fps is not None else vfps
fstp = ceil(vfps / fps)
nframes = int(fpc * fstp)
indices = np.arange(sf, sf + nframes, fstp)
```

Ý nghĩa:

- `vfps`: FPS thật của video gốc, ví dụ 30 FPS.
- `fps`: FPS mục tiêu lấy mẫu, config public dùng `fps: 4`.
- `fstp`: bước nhảy frame trong video gốc.
- `indices`: danh sách frame index được lấy cho clip.

Ví dụ nếu video gốc khoảng 30 FPS và target `fps=4`:

```text
vfps = 30
fps = 4
fstp = ceil(30 / 4) = 8
```

Khi đó source lấy frame kiểu:

```text
[100, 108, 116, 124, 132, 140, 148, 156]
```

Chứ không lấy:

```text
[100, 101, 102, 103, 104, 105, 106, 107]
```

Khoảng cách thời gian giữa hai frame cạnh nhau xấp xỉ:

```text
8 / 30 giây ≈ 0.267 giây
```

Gần với target:

```text
1 / 4 giây = 0.25 giây
```

Vì vậy khi nói source dùng `fps=4`, nghĩa là model nhìn một chuỗi frame đã được downsample theo thời gian, mỗi bước latent cách nhau khoảng 0.25 giây.

---

#### Config source public DROID có T=8

Trong `vjepa2/configs/train/vitg16/droid-256px-8f.yaml`:

```yaml
dataset_fpcs:
  - 8
fps: 4
```

Trong `train.py`:

```python
max_num_frames = max(dataset_fpcs)
```

Nên clip train có:

```text
T = 8 frame
```

Nhưng 8 frame này không phải 8 frame raw sát nhau. Chúng là 8 frame lấy theo target FPS.

Nếu target là 4 FPS, clip 8 frame bao phủ khoảng thời gian:

```text
(T - 1) / fps = (8 - 1) / 4 = 1.75 giây
```

Đó là lý do source có thể vừa dùng `T=8`, vừa không bị "frame quá sát nhau".

---

#### JEPA của bạn bạn giống source ở điểm nào?

JEPA của bạn bạn hiện dùng:

```yaml
horizon: 4
frame_stride: 2
```

Giả sử dữ liệu xe RC được lưu khoảng 9 FPS. Nếu lấy `frame_stride=2`, thì khoảng cách thời gian giữa hai latent steps là:

```text
2 / 9 giây ≈ 0.222 giây
```

Effective FPS tương đương:

```text
9 / 2 = 4.5 FPS
```

Con số này khá gần tinh thần source Meta:

```text
source Meta target fps = 4 FPS
JEPA bạn bạn effective fps ≈ 4.5 FPS
```

Vì vậy khi nói JEPA của bạn bạn gần tinh thần source hơn ở temporal spacing, ý là:

```text
cả hai đều cố tình làm các frame trong sample cách nhau đủ xa theo thời gian,
thay vì lấy mọi frame sát nhau.
```

Tuy nhiên cần nói chính xác:

- JEPA của bạn bạn giống source ở khoảng cách thời gian giữa hai latent steps.
- JEPA của bạn bạn không giống source ở số frame/sample, vì source public DROID dùng `T=8`, còn JEPA của bạn bạn dùng `horizon=4`.
- JEPA của bạn bạn cũng không giống source ở input pipeline, vì source đọc raw video, còn JEPA đọc precomputed feature cache.

---

#### NN-JEPA hiện tại khác ở điểm nào?

NN-JEPA hiện tại dùng:

```yaml
raw_frames_per_sample: 8
sequence_stride: 1
```

Trong code hiện tại, `sequence_stride` là bước trượt cửa sổ sample, không phải khoảng cách giữa frame bên trong sample. Vì vậy một sample đang lấy:

```text
[100, 101, 102, 103, 104, 105, 106, 107]
```

Nếu data manifest tương ứng khoảng 9 FPS, thì khoảng cách thời gian giữa hai frame cạnh nhau là:

```text
1 / 9 giây ≈ 0.111 giây
```

Effective FPS:

```text
9 FPS
```

So với source:

```text
source ≈ 4 FPS
NN-JEPA hiện tại ≈ 9 FPS nếu manifest giữ mọi frame ở 9 FPS
```

Nghĩa là NN-JEPA hiện tại có frame sát nhau hơn. Điều này có hai hệ quả:

1. Dễ học hơn về loss ngắn hạn vì frame sau giống frame trước hơn.
2. Có thể kém hơn cho dynamics/planning vì action effect chưa hiện rõ trong một bước.

---

#### Vì sao không chỉ đổi `sequence_stride=2` trong NN-JEPA?

Vì trong NN-JEPA hiện tại, `sequence_stride` chỉ đổi bước trượt cửa sổ, không đổi frame bên trong sample.

Ví dụ với `raw_frames_per_sample=8`, `sequence_stride=2`, code hiện tại tạo:

```text
sample 1 = [100, 101, 102, 103, 104, 105, 106, 107]
sample 2 = [102, 103, 104, 105, 106, 107, 108, 109]
sample 3 = [104, 105, 106, 107, 108, 109, 110, 111]
```

Nó không tạo:

```text
[100, 102, 104, 106, 108, 110, 112, 114]
```

Muốn giống source/JEPA ở temporal spacing, NN-JEPA cần thêm tham số mới, ví dụ:

```yaml
data:
  raw_frames_per_sample: 8
  frame_stride_inside_sample: 2
```

Khi đó sample mới là:

```text
[100, 102, 104, 106, 108, 110, 112, 114]
```

Nếu muốn giảm memory giống JEPA của bạn bạn:

```yaml
data:
  raw_frames_per_sample: 4
  frame_stride_inside_sample: 2
```

Sample:

```text
[100, 102, 104, 106]
```

---

#### Vậy nên chọn kiểu nào?

Nếu mục tiêu là gần source Meta public DROID hơn:

```yaml
raw_frames_per_sample: 8
frame_stride_inside_sample: khoảng sao cho effective fps ≈ 4
```

Ví dụ data manifest khoảng 9 FPS:

```yaml
raw_frames_per_sample: 8
frame_stride_inside_sample: 2
```

Khi đó effective FPS khoảng:

```text
9 / 2 = 4.5 FPS
```

Nếu mục tiêu là train thử nhanh/ít OOM hơn:

```yaml
raw_frames_per_sample: 4
frame_stride_inside_sample: 2
```

Cách này giống JEPA của bạn bạn hơn về memory và tốc độ, nhưng không còn giống source Meta ở số frame `T=8`.

Kết luận chính xác:

```text
NN-JEPA hiện tại giống source hơn ở T=8 và batch shuffle.
JEPA của bạn bạn giống source hơn ở việc không lấy frame quá sát nhau theo raw FPS.
Muốn NN-JEPA đúng hơn cả hai phía, nên thêm frame_stride_inside_sample thay vì chỉ đổi sequence_stride.
```

---

### 14.0.6. Có nên port SessionBatchSampler sang NN-JEPA không?

Có, nên port.

Nhưng nên hiểu đúng: đây là tối ưu DataLoader/I/O, không phải thay đổi bản chất loss JEPA-AC.

Nên implement dưới dạng option:

```yaml
dataloader:
  sampler: session_batch
```

hoặc:

```yaml
train:
  batch_sampler: session
```

Không nên xóa random shuffle toàn cục. Nên giữ cả hai mode:

```text
global_shuffle  = dễ hiểu, chuẩn ML phổ thông, debug đơn giản
session_batch   = ổn định RAM/I/O hơn cho feature cache lớn
```

Khi so sánh thí nghiệm, có thể chạy:

```text
same model + same data + global_shuffle
same model + same data + session_batch
```

Nếu metric tương đương nhưng session batch không OOM, thì dùng session batch cho train dài.

### Ưu tiên 1: thêm session-batch sampler cho feature dataset

Mục tiêu:

- batch chỉ chứa sample trong cùng session
- giữ memmap cache nóng
- giảm nhảy file
- giảm áp lực RAM/I/O

NN-JEPA có thể học theo `SessionBatchSampler` trong `JEPA/src/jepa_wm/data/ac_clip.py`.

### Ưu tiên 2: expose DataLoader knobs trong Hydra

Hiện NN-JEPA chỉ expose `num_workers`. Nên thêm:

```yaml
dataloader:
  pin_memory: false
  persistent_workers: false
  prefetch_factor: 2
  train_drop_last: true
  session_batch_sampler: true
```

Val/test nên có config riêng:

```yaml
eval_dataloader:
  num_workers: 0
  pin_memory: false
  persistent_workers: false
  prefetch_factor: null
```

### Ưu tiên 3: thêm bf16 autocast option

Nên thêm:

```yaml
train:
  amp_dtype: bf16
  use_amp: true
```

Trong train/eval:

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    outputs = compute_world_model_losses(...)
```

Lưu ý:

- bf16 thường ổn định hơn fp16.
- Loss vẫn có thể log bằng float32.
- Nếu GPU không hỗ trợ bf16 thì fallback fp32 hoặc fp16 phải kiểm tra kỹ.

### Ưu tiên 4: tạo feature cache fp16 hoặc bf16

NN-JEPA hiện đang có `fp32` cache rất lớn. Có thể thêm preset:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

Lợi ích:

- giảm disk khoảng 2 lần
- giảm RAM load từ DataLoader
- giảm bandwidth I/O

Rủi ro:

- feature precision giảm nhẹ
- cần kiểm tra val loss/rollout metric so với fp32
- nếu muốn tuyệt đối chính xác khoa học, giữ fp32 làm baseline và fp16 làm experiment nhanh/ổn định

### Ưu tiên 5: thêm config sample ngắn kiểu JEPA

Tạo experiment:

```yaml
data:
  raw_frames_per_sample: 4
  sequence_stride: 1
  frame_stride_inside_sample: 2   # tham số mới cần implement, hiện NN-JEPA chưa có
  auto_steps: 2
```

Lý do:

- gần cách JEPA đang dùng `horizon=4`, `frame_stride=2`
- giảm attention memory khoảng 4 lần so với 8 frame
- vẫn giữ được multi-step dynamics

Lưu ý quan trọng: chỉ đổi `sequence_stride: 2` trong NN-JEPA hiện tại **không** làm sample lấy frame cách nhau 2. Nó chỉ làm cửa sổ sample trượt thưa hơn. Muốn thật sự giống JEPA cần thêm logic lấy frame bên trong sample theo stride.

### Ưu tiên 6: giảm W&B overhead khi train đêm

Config ổn định hơn:

```yaml
wandb:
  watch_log: none
  grad_stats_every: 0
  param_stats_every: 0
  log_every: 50
```

Sau khi pipeline ổn định thì bật lại `grad_stats_every` để debug.

---

## 15. Đề Xuất Lộ Trình Sửa NN-JEPA

### Giai đoạn A: Chống OOM trước

Mục tiêu: train qua đêm không chết.

Thay đổi nên làm:

- val/test `num_workers=0` hoặc `2`
- tắt `persistent_workers`
- giảm `prefetch_factor` từ `4` xuống `2`
- tắt W&B watch/grad stats
- giữ `eval_batch_size=2`

### Giai đoạn B: Port session-batch sampler

Mục tiêu: DataLoader ổn định và nhanh hơn.

Thay đổi:

- Dataset windows lưu `(session_id, window_start)` rõ ràng.
- Sampler group windows theo session.
- Batch lấy cùng session.
- Có `set_epoch(epoch)` để shuffle ổn định.

### Giai đoạn C: Thêm bf16 autocast

Mục tiêu: giảm VRAM và tăng tốc.

Thay đổi:

- train/eval bọc `autocast`.
- checkpoint lưu config AMP.
- thêm kiểm tra GPU support.

### Giai đoạn D: Config JEPA-like

Mục tiêu: so sánh công bằng hơn với JEPA.

Tạo Hydra experiment:

```text
rc_jepa_tiny_newdata_jepa_like
```

Nội dung:

```yaml
data:
  raw_frames_per_sample: 4
  sequence_stride: 1
  frame_stride_inside_sample: 2   # cần thêm vào code dataset
  auto_steps: 2
model:
  type: simple hoặc official_lite
  size: tiny
train:
  batch_size: 24 hoặc 32
  eval_batch_size: 2
  num_workers: 2
```

### Giai đoạn E: Feature cache fp16

Mục tiêu: giảm disk/RAM/I/O.

Tạo extractor output:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

Sau đó train lại cùng config để so:

- fp32 baseline
- fp16 cache
- bf16 train
- horizon 4 stride 2

---

## 16. Những Điều Không Nên Làm Ngay

### 16.1. Không nên chỉ tăng predictor size

Tăng từ tiny lên base có thể cải thiện biểu diễn nhưng không giải quyết gốc OOM.

Nếu pipeline hiện tại còn OOM với tiny, thì base/official_lite nhiều khả năng còn khó ổn định hơn.

### 16.2. Không nên bật W&B watch all khi train dài

`watch_log=all` hoặc log gradient quá dày rất hữu ích khi debug, nhưng không phù hợp khi DataLoader đang sát RAM.

### 16.3. Không nên trộn feature/checkpoint JEPA và NN-JEPA

JEPA mới:

```text
ViT-L, D=1024, 576 token/frame
```

NN-JEPA hiện tại:

```text
ViT-B, D=768, 576 token/frame
```

Hai loại feature này không cùng chiều `D`, không thể dùng lẫn checkpoint predictor.

---

## 17. Kết Luận Kỹ Thuật

JEPA của bạn bạn ổn hơn vì thiết kế pipeline hợp lý hơn cho patch-token world model:

- Cache `fp16` thay vì `fp32`.
- Train/val `bf16` thay vì `fp32`.
- Sample `4 frame` thay vì `8 frame`.
- Batch theo session thay vì shuffle toàn cục.
- DataLoader ít aggressive hơn.
- W&B logging nhẹ hơn.
- Có `horizon=4` làm context ngắn hơn, và `frame_stride=2` làm transition giữa các latent steps có ý nghĩa chuyển động hơn.

NN-JEPA hiện tại đúng hướng về mặt ý tưởng JEPA-AC, nhưng pipeline đang thiên về an toàn/đầy đủ ban đầu hơn là tối ưu memory. Vì vậy dù predictor tiny rất nhỏ, RAM vẫn có thể nổ do feature batch + DataLoader + fp32 + sequence dài.

Khuyến nghị thực dụng nhất:

1. Sửa DataLoader trước.
2. Thêm session-batch sampler.
3. Thêm bf16 autocast.
4. Tạo config `horizon=4`, `stride=2`.
5. Sau đó mới thử official-lite/base hoặc feature ViT-L mạnh hơn.

Nếu làm đúng theo thứ tự này, NN-JEPA sẽ vừa gần JEPA của bạn bạn hơn, vừa giảm khả năng OOM khi train/val overnight.
