# Báo cáo: Triển khai `official_lite` predictor cho NN-JEPA

Ngày viết: 2026-06-09

Mục tiêu:

- Giữ nguyên `SimpleACPredictor` cũ để làm baseline.
- Thêm predictor mới theo hướng gần source V-JEPA AC hơn.
- Cho phép chọn predictor bằng tham số train.
- Không sửa source gốc trong `vjepa2/`.
- Đảm bảo train/eval/infer đọc đúng checkpoint theo loại predictor đã train.

## Kết luận ngắn

NN-JEPA hiện đã có hai loại predictor:

```text
simple
official_lite
```

Chọn bằng CLI:

```bash
--predictor-type simple
```

hoặc:

```bash
--predictor-type official_lite
```

`simple` là baseline cũ, không bị xóa.

`official_lite` là predictor mới, bám theo logic official V-JEPA AC:

- Token layout theo từng frame là `[action token, state token, patch tokens]`.
- Dùng action-block causal attention mask giống ý tưởng `build_action_block_causal_attention_mask` trong source `vjepa2`.
- Dùng RoPE attention theo source V-JEPA AC.
- Output vẫn là latent patch tokens, không dự đoán action trực tiếp.
- Loss vẫn dùng `teacher_forcing_loss + rollout_loss`.

## File đã thay đổi

Các file code chính:

```text
src/models/rc_jepa_ac.py
src/tools/train_rc_jepa_ac_features.py
src/tools/train_rc_jepa_ac.py
src/tools/rc_jepa_ac_feature_runtime.py
src/tools/infer_rc_jepa_ac_features.py
src/tools/train_rc_jepa_ac_features_hydra.py
src/models/__init__.py
```

Các file config/test:

```text
configs/hydra/experiment/rc_jepa_tiny.yaml
configs/hydra/experiment/rc_jepa_small.yaml
configs/hydra/experiment/rc_jepa_base.yaml
configs/hydra/experiment/rc_jepa_official_lite_tiny.yaml
tests/test_rc_jepa_ac.py
tests/test_hydra_train_config.py
```

## Predictor types hiện tại

Trong `src/models/rc_jepa_ac.py`:

```python
DEFAULT_PREDICTOR_TYPE = "simple"
SUPPORTED_PREDICTOR_TYPES = ("simple", "official_lite")
```

Factory mới:

```python
build_ac_predictor(...)
```

Factory này dựng một trong hai model:

- `SimpleACPredictor`
- `VJepaStyleACPredictor`

Điều này giúp train/eval/infer không hard-code `SimpleACPredictor` nữa.

## `SimpleACPredictor` hiện tại

`SimpleACPredictor` vẫn giữ nguyên vai trò baseline.

Luồng xử lý:

1. Nhận latent tokens `[B, T*K, D]`.
2. Reshape thành `[B, T, K, D]`.
3. Project latent/action/state sang `predictor_dim`.
4. Thêm learned `frame_pos`, `patch_pos`, `action_type`, `state_type`.
5. Ghép token theo frame:

```text
[action, state, patches]
```

6. Flatten thành sequence:

```text
[B, T * (K + 2), predictor_dim]
```

7. Chạy qua `nn.TransformerEncoder`.
8. Dùng time-causal mask đơn giản.
9. Bỏ action/state token, giữ patch tokens.
10. Project output về latent dim.

Ưu điểm:

- Dễ đọc.
- Dễ debug.
- Chạy nhanh hơn.
- Ít phụ thuộc logic phức tạp.
- Phù hợp baseline và smoke test.

## `official_lite` predictor mới

Class mới:

```python
VJepaStyleACPredictor
```

File:

```text
src/models/rc_jepa_ac.py
```

Ý tưởng:

`official_lite` giữ các phần quan trọng từ predictor official V-JEPA AC nhưng dùng size nhỏ để train được trên máy local.

Nó bám theo source public:

```text
vjepa2/src/models/ac_predictor.py
vjepa2/src/models/utils/modules.py
```

Các điểm bám source:

- Có `predictor_embed` để project latent encoder dim sang predictor dim.
- Có `action_encoder`.
- Có `state_encoder`.
- Ghép `[action, state, patch tokens]` theo từng frame.
- Dùng action-block causal attention mask.
- Dùng RoPE attention cho frame/height/width positions.
- Sau transformer block, bỏ action/state token.
- Output là predicted latent tokens.

Điểm adapt cho xe RC:

- Official DROID dùng action/state cùng dim 7.
- NN-JEPA dùng action 2D và state 5D.
- Vì vậy `official_lite` tách riêng:

```python
action_encoder = nn.Linear(action_dim, predictor_dim)
state_encoder = nn.Linear(state_dim, predictor_dim)
```

Điểm này là cần thiết để phù hợp dữ liệu RC.

## Token layout của `official_lite`

Với mỗi frame:

```text
[action token, state token, patch_1, patch_2, ..., patch_K]
```

Trong NN-JEPA ViT-B 384 hiện tại:

```text
K = 576
D = 768
T = 8 frame/sample
```

Một sample full:

```text
latent input = [B, T * K, D]
             = [B, 8 * 576, 768]
             = [B, 4608, 768]
```

Sau khi thêm action/state token:

```text
sequence length = T * (K + 2)
                = 8 * (576 + 2)
                = 4624 token
```

Đây là lý do `official_lite` cần batch nhỏ hơn `simple`.

## Action-block causal attention mask

Function mới:

```python
build_action_block_causal_attention_mask(...)
```

Ý nghĩa:

- Mỗi frame là một block token.
- Query ở frame `t` chỉ được attend tới frame `<= t`.
- Query ở frame quá khứ không được nhìn token frame tương lai.
- Mask trả về theo semantics của `torch.nn.functional.scaled_dot_product_attention`: `True` nghĩa là được attend.

Ví dụ với 3 frame:

```text
frame 0 được nhìn: frame 0
frame 1 được nhìn: frame 0, frame 1
frame 2 được nhìn: frame 0, frame 1, frame 2
```

Đây là khác biệt quan trọng so với mask đơn giản cũ. Nó sát cách source V-JEPA AC build attention theo block action/state/patch hơn.

## RoPE attention

`official_lite` có attention riêng:

```python
VJepaStyleACAttention
```

Nó xử lý:

- Action/state tokens.
- Patch tokens.
- Frame position.
- Height position.
- Width position.

RoPE được adapt từ source public. Một chi tiết quan trọng: source public có pattern lặp frequency riêng để tương thích checkpoint. Bản `official_lite` giữ kiểu lặp đó, nhưng làm robust hơn cho predictor nhỏ để không vỡ khi chiều RoPE rất nhỏ.

## Tham số thực tế

Đã đếm trực tiếp bằng code hiện tại với cấu hình feature ViT-B 384:

```text
latent_dim = 768
tokens_per_frame = 576
state_dim = 5
action_dim = 2
max_frames = 8
```

| Predictor | Size | `predictor_dim` | `depth` | `heads` | Trainable params |
|---|---:|---:|---:|---:|---:|
| `simple` | `tiny` | 128 | 2 | 4 | 670,464 |
| `simple` | `small` | 256 | 4 | 4 | 3,706,112 |
| `simple` | `base` | 512 | 6 | 8 | 20,007,680 |
| `official_lite` | `tiny` | 128 | 2 | 4 | 595,456 |
| `official_lite` | `small` | 256 | 4 | 4 | 3,556,096 |
| `official_lite` | `base` | 512 | 6 | 8 | 19,707,648 |

Lưu ý:

- `official_lite` không nhất thiết nhiều tham số hơn `simple`.
- Nhưng `official_lite` vẫn có thể nặng hơn khi chạy vì attention/RoPE/mask phức tạp hơn.
- Chi phí lớn nhất đến từ sequence length `4624 token/sample`, không chỉ từ số tham số.

## CLI train từ feature cache

Script chính:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features
```

Chạy baseline cũ:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
    --predictor-type simple \
    --model-size tiny
```

Chạy official-lite tiny:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
    --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
    --manifest-dir data/processed/manifests \
    --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny \
    --predictor-type official_lite \
    --model-size tiny \
    --epochs 20 \
    --batch-size 4 \
    --eval-batch-size 1 \
    --num-workers 8 \
    --lr 1e-4 \
    --warmup-epochs 2 \
    --warmup-start-factor 0.1 \
    --min-lr-ratio 0.1 \
    --early-stopping-patience 5
```

Khuyến nghị ban đầu:

- Dùng `official_lite tiny`.
- Batch size nên bắt đầu từ `4`.
- Eval batch size nên bắt đầu từ `1`.
- Nếu không OOM mới tăng dần.

## Hydra config mới

Config mới:

```text
configs/hydra/experiment/rc_jepa_official_lite_tiny.yaml
```

Chạy:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
    experiment=rc_jepa_official_lite_tiny
```

Dry run:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
    experiment=rc_jepa_official_lite_tiny \
    runtime.dry_run=true \
    runtime.require_cuda=false \
    wandb.disabled=true \
    train.device=cpu
```

Dry run đã kiểm tra ra đúng:

```text
predictor_type = official_lite
model_size = tiny
predictor_dim = 128
predictor_depth = 2
predictor_heads = 4
batch_size = 4
eval_batch_size = 1
```

## Eval và inference

Runtime helper đã được sửa để đọc `predictor_type` từ checkpoint.

File:

```text
src/tools/rc_jepa_ac_feature_runtime.py
```

Điều này nghĩa là:

- Checkpoint `simple` sẽ dựng lại `SimpleACPredictor`.
- Checkpoint `official_lite` sẽ dựng lại `VJepaStyleACPredictor`.

Eval:

```bash
PYTHONPATH=src python3 -m tools.eval_rc_jepa_ac_features \
    --checkpoint checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny/best.pt \
    --eval-batch-size 1
```

Inference:

```bash
PYTHONPATH=src python3 -m tools.infer_rc_jepa_ac_features \
    --checkpoint checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny/best.pt \
    --eval-batch-size 1 \
    --max-samples 32
```

## Resume

Train checkpoint lưu `args`, trong đó có:

```text
predictor_type
predictor_dim
predictor_depth
predictor_heads
dropout
```

Khi resume, script kiểm tra các field này. Nếu checkpoint được train bằng `official_lite` nhưng lệnh resume lại dùng `simple`, script sẽ báo lỗi rõ ràng.

Ví dụ resume official-lite:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
    --resume-from checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny/last.pt \
    --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny \
    --predictor-type official_lite \
    --model-size tiny \
    --batch-size 4 \
    --eval-batch-size 1
```

Nếu dùng Hydra:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
    experiment=rc_jepa_official_lite_tiny \
    train.resume_from=checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny/last.pt
```

## Vì sao không import trực tiếp `vjepa2/src/models/ac_predictor.py`?

Không import trực tiếp vì:

- Repo NN-JEPA cũng có thư mục `src`.
- Repo `vjepa2` cũng có thư mục `src`.
- Import package dạng `src.models...` dễ xung đột namespace.
- User đã yêu cầu không đụng code gốc trong `vjepa2/`.

Vì vậy cách an toàn hơn là:

- Đọc source official.
- Port/adapt phần cần thiết vào `src/models/rc_jepa_ac.py`.
- Giữ code tự chứa trong NN-JEPA.
- Không sửa `vjepa2/`.

## Điểm giống official V-JEPA AC

`official_lite` giống ở các điểm quan trọng:

- Dùng latent token-level.
- Không mean-pool latent.
- Ghép action/state token vào từng frame.
- Dùng action-block causal attention mask.
- Dùng RoPE attention logic cho token vị trí frame/height/width.
- Output là latent tokens.
- Dùng teacher forcing và autoregressive rollout qua loss hiện tại.

## Điểm khác official V-JEPA AC

Vẫn có khác biệt, cần ghi rõ:

- Official public AC dùng DROID action/state 7D, NN-JEPA dùng RC action 2D/state 5D.
- Official config dùng predictor rất lớn: depth 24, dim 1024, heads 16.
- `official_lite` dùng size nhỏ: tiny/small/base để chạy local.
- Official AC public checkpoint là V-JEPA 2 ViT-g/16 256.
- NN-JEPA hiện dùng V-JEPA 2.1 ViT-B/16 384 feature cache.
- Official train source có nhiều distributed/mixed precision/activation checkpointing hơn.

Vì vậy `official_lite` là official-style predictor, không phải exact official checkpoint reproduction.

## Rủi ro khi dùng `official_lite`

Các rủi ro chính:

- Có thể OOM nhanh hơn `simple`.
- Có thể train chậm hơn.
- Có thể cần batch size nhỏ hơn.
- Eval nên để batch size `1` trước.
- Nếu data chưa đủ đa dạng, predictor mạnh hơn có thể overfit.
- Nếu feature cache còn thiếu session, train vẫn fail như cũ.

## Kết quả kiểm tra

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m compileall -q src tests
```

Kết quả: pass.

Đã chạy:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m unittest discover -s tests -v
```

Kết quả:

```text
Ran 41 tests
OK
```

Đã chạy Hydra dry-run cho config mới:

```bash
conda run -n nn-jepa env PYTHONPATH=src python -m tools.train_rc_jepa_ac_features_hydra \
    experiment=rc_jepa_official_lite_tiny \
    runtime.dry_run=true \
    runtime.require_cuda=false \
    wandb.disabled=true \
    train.device=cpu
```

Kết quả: pass, args map đúng sang `predictor_type=official_lite`.

Đã chạy smoke loss với `official_lite`:

```text
loss scalar
teacher_forcing_loss scalar
rollout_loss scalar
```

## Khuyến nghị chạy thực tế

Thứ tự nên làm:

1. Chạy `official_lite tiny`, batch size `4`, eval batch size `1`.
2. Nếu OOM, giảm batch size về `2` hoặc `1`.
3. So sánh với `simple tiny` trên cùng split.
4. Nếu `official_lite tiny` thắng val/test ổn định, thử `official_lite small`.
5. Không chạy `official_lite base` ngay nếu chưa biết VRAM peak.

Lệnh nên thử đầu tiên:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features_hydra \
    experiment=rc_jepa_official_lite_tiny
```

Nếu muốn CLI thuần:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
    --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
    --manifest-dir data/processed/manifests \
    --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_official_lite_tiny \
    --predictor-type official_lite \
    --model-size tiny \
    --epochs 20 \
    --batch-size 4 \
    --eval-batch-size 1 \
    --num-workers 8 \
    --lr 1e-4 \
    --warmup-epochs 2 \
    --warmup-start-factor 0.1 \
    --min-lr-ratio 0.1 \
    --early-stopping-patience 5
```

## Kết luận cuối

NN-JEPA hiện đã hỗ trợ predictor mới theo hướng official-lite mà không xóa baseline cũ. Đây là bước tiến hợp lý để tiến gần V-JEPA AC official hơn, nhưng vẫn giữ workflow thực dụng: có baseline đơn giản, có model nhỏ để thử nhanh, có checkpoint/eval/infer tương thích, và có test bảo vệ các lỗi shape/mask cơ bản.
