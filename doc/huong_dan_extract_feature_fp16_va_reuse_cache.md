# Hướng dẫn extract feature fp16 và reuse cache

Ngày viết: 2026-06-10

Phạm vi: repo `NN-JEPA`. File này ghi lại quy trình extract feature V-JEPA 2.1 ViT-B 384 cho hai experiment chính:

- Baseline/newdata: chỉ dùng data hiện tại, không trộn servo cũ.
- Mixed old-servo: dùng data hiện tại + data servo cũ trong `servo_old_mix_v1`.

Mục tiêu là tránh nhầm folder cache, tránh extract lại vô ích, và biết cách recover nếu lỡ chạy sai lệnh.

## 1. Khái niệm cần nhớ

Feature cache là output của encoder V-JEPA frozen.

Pipeline hiện tại:

```text
ảnh processed
-> V-JEPA 2.1 ViT-B/16 384 frozen encoder
-> latent token mỗi frame
-> lưu .npy theo từng session
-> train predictor/world model từ feature cache
```

Với preset hiện tại:

```text
encoder_preset = vitb_384
encoder_name = vit_base_384
checkpoint_key = ema_encoder
image_size = 384
patch_size = 16
tokens_per_frame = 576
embed_dim = 768
dtype lưu trên disk = fp16
```

Một frame sau khi encode có shape:

```text
576 token x 768 dim
```

Nếu lưu `fp16`, một frame tốn khoảng:

```text
576 * 768 * 2 bytes ~= 0.84 MiB
```

## 2. Vì sao không dùng chung một folder cache cho mọi experiment

Không nên dùng chung cùng một thư mục cache cho baseline và mixed.

Lý do: mỗi feature cache folder có một file:

```text
metadata.json
```

File này mô tả:

```text
manifest_dir
splits
session_count
frame_count
encoder_name
checkpoint_key
dtype
tokens_per_frame
embed_dim
```

Baseline và mixed dùng hai manifest khác nhau:

```text
baseline manifest:
data/processed/manifests

mixed manifest:
data/experiments/servo_old_mix_v1/processed/manifests
```

Vì vậy chúng phải có hai folder cache riêng:

```text
baseline feature cache:
data/processed/features/vjepa2_1_vitb_384_ema_fp16

mixed old-servo feature cache:
data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
```

Điểm quan trọng: không dùng chung folder, nhưng có thể reuse file feature theo session bằng `--seed-from-features-dir`.

## 3. Reuse cache là gì?

Nếu cùng một session, cùng frame, cùng encoder, cùng checkpoint, cùng dtype, thì feature là giống nhau.

Vì vậy có thể reuse từ cache này sang cache khác.

Tool dùng tham số:

```bash
--seed-from-features-dir <folder-cache-nguồn>
```

Khi chạy, tool sẽ:

1. Đọc manifest đích.
2. Kiểm tra session nào cần feature.
3. Tìm session đó trong cache nguồn.
4. Nếu `.npy + .json` tồn tại và metadata khớp, tool tạo symlink sang cache đích.
5. Nếu thiếu hoặc không khớp shape/dtype, tool mới encode lại session đó.

Metadata bắt buộc khớp:

```text
format_version
feature_layout
encoder_name
checkpoint_key
image_size
patch_size
tubelet_size
tokens_per_frame
embed_dim
dtype
normalization_mean
normalization_std
checkpoint_path
```

Các trường như `manifest_dir`, `splits`, `session_count`, `frame_count` có thể khác giữa baseline và mixed, vì đây là hai experiment khác nhau.

## 4. Quy trình khuyến nghị nếu làm từ đầu

Nếu chưa có cache nào, nên extract baseline trước, rồi extract mixed bằng cách seed từ baseline.

### 4.1. Extract baseline current data

Lệnh:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 256 \
  --num-workers 16 \
  --dtype fp16
```

Ý nghĩa:

- Đọc manifest baseline từ `data/processed/manifests`.
- Ghi cache baseline vào `data/processed/features/vjepa2_1_vitb_384_ema_fp16`.
- Không có data servo cũ.
- Dùng encoder V-JEPA 2.1 ViT-B 384.
- Lưu feature dạng `fp16`.

Sau lệnh này, baseline train config có thể dùng:

```text
features_dir = data/processed/features/vjepa2_1_vitb_384_ema_fp16
manifest_dir = data/processed/manifests
```

### 4.2. Extract mixed old-servo bằng seed từ baseline

Lệnh:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --output-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --seed-from-features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 256 \
  --num-workers 16 \
  --dtype fp16 \
  --splits train val
```

Ý nghĩa:

- Đọc manifest mixed từ `data/experiments/servo_old_mix_v1/processed/manifests`.
- Ghi cache mixed vào `data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16`.
- Reuse feature của current sessions đã có trong baseline cache.
- Chỉ encode thêm session old-servo hoặc session còn thiếu.
- Chỉ dùng `train val` vì `test` hiện là alias của `val`.

Sau lệnh này, mixed train config có thể dùng:

```text
features_dir = data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
```

## 5. Quy trình nếu đã extract mixed trước

Nếu mixed cache đã có trước baseline cache, có thể seed ngược lại.

Lệnh tạo baseline cache từ mixed cache:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/processed/manifests \
  --output-dir data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
  --seed-from-features-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 256 \
  --num-workers 16 \
  --dtype fp16
```

Ý nghĩa:

- Đích là baseline cache.
- Nguồn seed là mixed cache.
- Tool sẽ reuse current sessions có trong mixed cache.
- Session nào thiếu thì mới encode.

Đây là cách recover tốt nếu lỡ extract mixed trước.

## 6. Ý nghĩa từng tham số quan trọng

`PYTHONPATH=src`

Cho Python thấy package trong `src/`.

`python3 -m tools.extract_vjepa_features`

Chạy tool extract feature.

`--vjepa-root vjepa2`

Đường dẫn tới repo V-JEPA 2.1 local.

`--encoder-preset vitb_384`

Chọn preset encoder:

```text
vitb_384 = V-JEPA 2.1 ViT-B/16, image 384, checkpoint ema_encoder
```

Đây là preset nhẹ nhất đang dùng cho pipeline này.

`--manifest-dir ...`

Chỉ định manifest cần extract.

Baseline:

```text
data/processed/manifests
```

Mixed:

```text
data/experiments/servo_old_mix_v1/processed/manifests
```

`--output-dir ...`

Chỉ định nơi lưu feature cache.

Luôn nên ghi rõ tham số này để tránh nhầm folder.

`--seed-from-features-dir ...`

Cache nguồn dùng để reuse session feature.

Không bắt buộc, nhưng nên dùng khi cache nguồn đã có.

`--batch-size 256`

Số frame encode trong một batch.

Batch lớn hơn thường nhanh hơn nếu VRAM đủ.

Với GPU 16GB hiện tại, `256` đang chạy được. Nếu muốn thử tối ưu:

```text
256 -> an toàn hiện tại
384 -> thử nếu GPU còn rảnh
512 -> chỉ thử nếu VRAM còn dư rõ
```

`--num-workers 16`

Số worker đọc ảnh, decode PIL, convert tensor, normalize.

Với máy 32GB RAM, 20 core/20 thread:

```text
8  = an toàn
12 = thường hợp lý
16 = đang dùng được
20 = chỉ thử nếu 16 vẫn nghẽn và disk/RAM còn dư
```

`--dtype fp16`

Lưu feature trên disk bằng float16.

Lợi ích:

- Giảm khoảng một nửa dung lượng so với fp32.
- Ít áp lực disk/RAM hơn.

Đánh đổi:

- Có sai số lượng tử hóa nhỏ so với fp32.

Hiện project đã chuyển default sang fp16 vì fp32 quá nặng.

`--splits train val`

Chỉ extract các split train và val.

Với mixed experiment hiện tại, `test.jsonl` chỉ là alias của `val.jsonl`, nên không cần extract test riêng.

## 7. Cách nhận biết đang chạy đúng

Khi chạy mixed đúng, metadata in ra phải có:

```text
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
splits = ["train", "val"]
session_count = 211
dtype = fp16
```

Và cache phải nằm ở:

```text
data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
```

Khi chạy baseline đúng, metadata phải có:

```text
manifest_dir = data/processed/manifests
dtype = fp16
```

Và cache phải nằm ở:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

## 8. Vì sao progress nhìn như chạy lại từ đầu?

Tool duyệt qua toàn bộ session list.

Ngay trước mỗi session, nó in:

```text
Extracting features: session_xxx
```

Dòng này có thể gây hiểu nhầm. Sau đó tool mới kiểm tra:

- Nếu `.npy + .json` đã đủ và shape/dtype khớp, nó skip rất nhanh.
- Nếu thiếu hoặc đang dở, nó encode session đó.

Vì vậy progress bar có thể bắt đầu từ `0/211`, nhưng không có nghĩa là encode lại tất cả.

Nếu thấy nhiều session nhảy rất nhanh trong vài giây, đó là skip cache.

## 9. Lỗi dễ gặp: extract mixed nhầm vào baseline path

Lỗi xảy ra khi chạy mixed nhưng quên `--output-dir`.

Ví dụ lệnh sai:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --batch-size 256 \
  --num-workers 16 \
  --dtype fp16 \
  --splits train val
```

Với code cũ, lệnh này có thể ghi mixed cache vào default baseline path:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

Khi đó `metadata.json` trong baseline path sẽ ghi:

```text
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
session_count = 211
```

Đó là dấu hiệu cache bị đặt nhầm chỗ.

Code hiện tại đã thêm guard: nếu manifest nằm trong `data/experiments/.../processed/manifests` và không truyền `--output-dir`, output mặc định sẽ tự chuyển sang:

```text
data/experiments/.../features/...
```

Dù vậy vẫn nên ghi rõ `--output-dir` trong lệnh để không nhầm.

## 10. Cách recover nếu extract mixed nhầm vào baseline path

Nếu cache baseline path thực chất là mixed cache, làm như sau.

### 10.1. Dừng process

Dùng:

```text
Ctrl+C
```

### 10.2. Move cache nhầm sang đúng mixed path

Lệnh:

```bash
mkdir -p data/experiments/servo_old_mix_v1/features
mv data/processed/features/vjepa2_1_vitb_384_ema_fp16 \
   data/experiments/servo_old_mix_v1/features/
```

Sau lệnh này:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16
```

sẽ không còn.

Mixed cache sẽ nằm đúng ở:

```text
data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16
```

### 10.3. Chạy lại mixed extract đúng output

Lệnh:

```bash
PYTHONPATH=src python3 -m tools.extract_vjepa_features \
  --vjepa-root vjepa2 \
  --encoder-preset vitb_384 \
  --manifest-dir data/experiments/servo_old_mix_v1/processed/manifests \
  --output-dir data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16 \
  --batch-size 256 \
  --num-workers 16 \
  --dtype fp16 \
  --splits train val
```

Không thêm `--overwrite`.

Tool sẽ:

- skip session đã complete;
- ghi lại session đang dở;
- encode tiếp session còn thiếu.

## 11. Không dùng `--overwrite` nếu không có lý do rõ ràng

Không dùng:

```bash
--overwrite
```

trừ khi thật sự muốn xóa/recompute logic theo session.

Lý do: khi không có `--overwrite`, tool sẽ skip session đã có cache đúng. Đây là cơ chế resume quan trọng.

Nếu thêm `--overwrite`, tool sẽ ghi lại session thay vì tận dụng cache cũ.

## 12. Lệnh audit cache nhanh

Kiểm tra số `.npy`, `.json`, và session complete:

```bash
python3 - <<'PY'
from pathlib import Path

for root in [
    Path("data/processed/features/vjepa2_1_vitb_384_ema_fp16/sessions"),
    Path("data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16/sessions"),
]:
    print(root)
    if not root.exists():
        print({"exists": False})
        continue
    npy = {p.stem for p in root.glob("*.npy")}
    js = {p.stem for p in root.glob("*.json")}
    print({
        "exists": True,
        "npy": len(npy),
        "json": len(js),
        "complete_pairs": len(npy & js),
        "npy_without_json": sorted(npy - js)[:10],
        "json_without_npy": sorted(js - npy)[:10],
    })
PY
```

Kiểm tra metadata:

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in [
    Path("data/processed/features/vjepa2_1_vitb_384_ema_fp16/metadata.json"),
    Path("data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16/metadata.json"),
]:
    print(path)
    if not path.exists():
        print({"exists": False})
        continue
    metadata = json.loads(path.read_text())
    print({
        "manifest_dir": metadata.get("manifest_dir"),
        "splits": metadata.get("splits"),
        "session_count": metadata.get("session_count"),
        "frame_count": metadata.get("frame_count"),
        "dtype": metadata.get("dtype"),
        "tokens_per_frame": metadata.get("tokens_per_frame"),
        "embed_dim": metadata.get("embed_dim"),
    })
PY
```

## 13. Lệnh train sau khi extract xong

### 13.1. Train baseline tiny newdata

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_newdata
```

### 13.2. Train mixed simple tiny

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_tiny_mix_oldservo_frame_stride2
```

### 13.3. Train mixed official-lite tiny

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
  experiment=rc_jepa_official_lite_tiny_mix_oldservo_frame_stride2
```

Config official-lite tiny mixed hiện nằm ở:

```text
configs/hydra/experiment/rc_jepa_official_lite_tiny_mix_oldservo_frame_stride2.yaml
```

## 14. Checklist trước khi bật train qua đêm

Kiểm tra cache baseline nếu train newdata:

```text
data/processed/features/vjepa2_1_vitb_384_ema_fp16/metadata.json exists
manifest_dir = data/processed/manifests
dtype = fp16
```

Kiểm tra cache mixed nếu train mixed:

```text
data/experiments/servo_old_mix_v1/features/vjepa2_1_vitb_384_ema_fp16/metadata.json exists
manifest_dir = data/experiments/servo_old_mix_v1/processed/manifests
splits = train, val
session_count = 211
dtype = fp16
```

Không được có tình trạng:

```text
baseline cache path nhưng metadata manifest_dir lại là servo_old_mix_v1
```

Nếu có, nghĩa là cache đang đặt nhầm chỗ. Xem mục recover ở trên.
