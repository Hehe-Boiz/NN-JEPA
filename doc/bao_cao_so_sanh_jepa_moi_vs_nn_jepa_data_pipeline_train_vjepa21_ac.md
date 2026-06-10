# Báo cáo so sánh JEPA mới vs NN-JEPA hiện tại về data pipeline và train V-JEPA 2.1 AC

Ngày kiểm tra: 2026-06-10

Workspace kiểm tra: `/home/heheboiz/data/NN-JEPA`

Repo JEPA được kiểm tra tại: `JEPA/`

Repo NN-JEPA được kiểm tra tại: thư mục gốc hiện tại.

## 1. Kết luận ngắn

JEPA của bạn bạn hiện đã đi xa hơn NN-JEPA ở phần robot stack thực chiến. Nó không chỉ train predictor nữa mà đã có sync dữ liệu, encode patch-token, train V-JEPA 2.1 AC cho xe RC, CEM planner, route planner web, topological graph, policy prior warm-start và closed-loop inference qua phone/ESP32.

NN-JEPA hiện vẫn là pipeline train/eval/research sạch hơn: manifest rõ ràng, metadata feature cache đầy đủ, Hydra, W&B logging, resume tốt hơn, test/val kiểm soát rõ hơn. Nhưng NN-JEPA đang thua JEPA ở tối ưu RAM/VRAM/I/O và ở mức tích hợp robot thật.

Điểm quan trọng nhất cần chú ý: NN-JEPA hiện đang extract feature từ `frame_path` đã preprocess resize `224x224`, rồi encoder nội suy lên `384x384`. JEPA đọc raw frame trực tiếp rồi resize lên `384x384` khi encode. Nếu muốn NN-JEPA bám V-JEPA 2.1 hơn, nên sửa extractor để ưu tiên `source_frame_path`.

Sau khi kiểm tra lại source kỹ hơn, có một điểm dễ hiểu sai nhưng rất quan trọng: trong `vjepa2/app/vjepa_droid/train.py`, DataLoader được gọi với `tubelet_size=1`, nên sample DROID vẫn lấy đủ `dataset_fpcs=8` frame thật. Sau đó train loop duplicate từng frame thành clip 2-frame trước khi đưa vào `target_encoder`, để encoder video có `tubelet_size=2` vẫn sinh token cho từng frame quan sát. Vì vậy `tubelet_size=2` trong source không có nghĩa là 8 frame bị gộp thành chỉ 4 bước action/state ở train AC.

## 2. Trạng thái repo JEPA sau khi kiểm tra

Repo JEPA hiện ở commit:

```text
f22990c HANDOFF: retrain 384 ep0-4 progress (ratio@1 0.816 @ep4 > v1 final) + LN mismatch quantified 0.58% (no restart needed)
```

`git status` trong `JEPA/` chỉ thấy:

```text
D data/.gitkeep
```

Tức là code chính không có thay đổi local chưa commit, ngoài file `.gitkeep` trong data bị xóa do data local.

## 3. JEPA mới đã cập nhật những gì

| Nhóm cập nhật | JEPA mới có gì | File chính |
|---|---|---|
| Sync dữ liệu | Re-pair frame với action/IMU bằng nội suy theo `t_scene_ms`, bù `dcam_ms`, xuất `actions_synced.csv` và `imu_synced.csv` | `JEPA/src/jepa_wm/data/sync.py` |
| Patch feature cache | Encode raw frame bằng V-JEPA 2.1 ViT-L 384, lưu full patch tokens fp16 `(N, 576, 1024)` | `JEPA/src/jepa_wm/engine/encode_patch.py` |
| V-JEPA 2.1 AC car model | `VJEPA2ACCar`, patch-token predictor, action/state token, block-causal transformer | `JEPA/src/jepa_wm/models/vjepa2_ac_car.py` |
| Multi-servo training | Hỗ trợ KDS + TowerPro bằng `roots`, thêm `domain_id` vào action token | `JEPA/configs/train/vjepa_ac_car.yaml` |
| Session-batch sampler | Batch chỉ chứa sample cùng session để giữ cache nóng, giảm reload session `.npy` lớn | `JEPA/src/jepa_wm/data/ac_clip.py` |
| Frozen split | Lưu split train/val vào `split.json`, lần sau reuse để val cố định | `JEPA/src/jepa_wm/engine/train_ac_car.py` |
| Train loop | BF16 autocast, gradient checkpointing, cosine warmup, early stopping, save `last.pt`/`best.pt` | `JEPA/src/jepa_wm/engine/train_ac_car.py` |
| CEM planner | Per-dim sigma, warm-start sigma theo policy prior, domain token, prev-action state | `JEPA/src/jepa_wm/planning/cem.py` |
| Route planner web | Web 2D map, Dijkstra, chọn waypoint/subgoal, lưu route, live status | `JEPA/scripts/route_web.py` |
| Closed-loop inference | Phone camera stream -> V-JEPA encode -> graph/subgoal -> CEM -> action về phone/ESP32 | `JEPA/scripts/inference_loop.py` |
| Policy prior | Behavior cloning policy để warm-start CEM theo tinh thần PiJEPA | `JEPA/scripts/train_policy_prior.py` |

## 4. So sánh data pipeline

Mốc đối chiếu với source Meta là public `vjepa2` ở các file chính: `vjepa2/configs/train/vitg16/droid-256px-8f.yaml`, `vjepa2/app/vjepa_droid/droid.py`, `vjepa2/app/vjepa_droid/train.py`. Lưu ý quan trọng: source public này là pipeline V-JEPA AC cho DROID, không phải một config “V-JEPA 2.1 AC cho xe RC” hoàn chỉnh.

| Phần | Source `vjepa2` public | JEPA của bạn bạn | NN-JEPA hiện tại | Cách nào chuẩn hơn |
|---|---|---|---|---|
| Raw data layout | DROID/video trajectory, loader trả `clips`, `actions`, `states`, `extrinsics` | `data/raw_towerpro/session_*`, `data/raw_kds/session_*`; mỗi session có `frames/`, `actions.csv`, `telemetry.csv`, `gyro.csv`, `accel.csv`, `rotvec.csv`, `gps.csv` | `data/raw/session_*`; mixed experiment lấy thêm old servo từ `JEPA/data/drive_extra_nonzip/...` | Với Meta exact: source. Với xe RC: JEPA/NN-JEPA đều custom hợp lý |
| Sync action/cảm biến | Dataset DROID đã có action/state align theo video index | Chạy `scripts/sync_dataset.py`; tạo `actions_synced.csv` và `imu_synced.csv` bằng nội suy theo `t_scene_ms` | Preprocess ưu tiên đọc `actions_synced.csv` và `imu_synced.csv`; nếu thiếu thì fallback raw sensor nearest-time | Với raw RC: JEPA chuẩn hơn vì có bước sync rõ ràng trước train |
| Bù camera delay | Không thấy logic `dcam_ms` kiểu RC trong source DROID | Nếu data cũ không có `dcam_ms` thì trừ mặc định 100 ms; data mới có `dcam_ms` thì offset 0 | Không tự chạy logic sync kiểu JEPA, chỉ đọc file đã sync hoặc match sensor theo tolerance | JEPA chuẩn hơn cho data RC từ app JEPA |
| Lọc frame | Loader lấy window hợp lệ từ video, kiểm soát video đủ dài | Sync bỏ frame ngoài telemetry, gap telemetry lớn, hoặc `mode != 1` | Bỏ thiếu frame/action, action ngoài range, duplicate frame, outlier robust theo state | JEPA mạnh ở sync; NN-JEPA mạnh ở manifest/report |
| Preprocess ảnh | Transform raw video frame trực tiếp theo `crop_size: 256` trong config DROID | Không cần ghi ảnh processed cho patch encoder; encode đọc raw `frames/*.jpg` rồi resize trực tiếp 384 | Preprocess ảnh sang `data/processed/images/...` resize 224x224 | JEPA gần source hơn. NN-JEPA đang có rủi ro 224 -> 384 khi extract |
| Feature extractor input | Train loop encode `clips` trực tiếp bằng `target_encoder` | Raw frame -> resize trực tiếp 384 | `sample["frame_path"]`, hiện là processed image 224 | Source exact là on-the-fly encoder. Trong cache offline, JEPA gần source hơn NN-JEPA |
| Feature cache format | Không lưu feature cache `.npy`; target feature được tính trong train | Mỗi session là một file `.npy`: `(N, Ntok, D)` fp16 | Có `metadata.json` + `sessions/session.npy` + `sessions/session.json` | Source exact không cache. Nếu cache offline thì NN-JEPA quản lý metadata chuẩn hơn |
| Feature cache dtype | Train dùng `dtype: bfloat16` | fp16 | fp16 cho cache hiện tại, fp32 đã xóa | JEPA gần tinh thần nhẹ bộ nhớ hơn; NN-JEPA rõ metadata hơn |
| Encoder chính | Config DROID public dùng `vit_giant_xformers`, checkpoint `vitg.pt`, `crop_size: 256` | V-JEPA 2.1 ViT-L 384 | V-JEPA 2.1 ViT-B 384 theo config hiện tại | Source exact là ViT-g 256. Trong V-JEPA 2.1 384, JEPA mạnh hơn NN-JEPA |
| Token/frame | Với `crop_size=256`, `patch_size=16` thì `tokens_per_frame=256`; target loop flatten thành `8*256` token | 576 token/frame ở 384px | 576 token/frame ở 384px | Source public DROID khác resolution. JEPA và NN-JEPA cùng 384-token layout |
| Embed dim | `vit_giant_xformers` encoder embed dim 1408; `pred_embed_dim=1024` là hidden dim predictor | 1024 với ViT-L 384 | 768 với ViT-B 384 | JEPA gần source hơn về predictor hidden/latent size, nhưng source latent exact là 1408 |
| Sample train | `dataset_fpcs: 8`, `fps: 4`, `auto_steps: 2`; DataLoader lấy đủ 8 frame, action length 7 | `horizon=4`, `frame_stride=2`, shape mỗi sample `(4,576,1024)` | `raw_frames_per_sample=8`, `frame_stride=2`, shape mỗi sample flatten `(8*576,768)` | NN-JEPA gần source hơn về T=8; JEPA gần source hơn nếu xét tốc độ train/VRAM |
| Tubelet trong train AC | `tubelet_size=2` thuộc encoder/predictor; train loop duplicate từng frame thành clip 2-frame để vẫn có token từng frame | Encode từng frame độc lập bằng image/video encoder offline | Encode từng frame độc lập bằng frozen encoder offline | Cả JEPA/NN-JEPA cache offline là lệch source, nhưng đúng nếu encoder frozen và transform cố định |
| Temporal spacing | Source lấy 8 frame theo target `fps=4`, ví dụ video 30 FPS sẽ lấy thưa frame theo `ceil(vfps/fps)` | Lấy `i + j*frame_stride`, thường `frame_stride=2` | Build từ manifest theo session, lấy `frame_stride` hoặc `target_fps` | NN-JEPA có `target_fps` gần source nhất; JEPA `frame_stride=2` gần source nếu data khoảng 8-10 FPS |
| Gap filtering | Video window phải đủ dài/hợp lệ | Có `max_gap` để bỏ window qua lỗ frame drop | Có `AC_MAX_FRAME_INDEX_GAP` và `AC_MAX_TIME_GAP_SEC` | Cả hai repo RC đều cần hơn source vì data RC dễ drop frame |
| Split train/val | Source dùng dataset path/config và `DistributedSampler` | Lần đầu train tự tạo `split.json` cạnh checkpoint, lần sau freeze | Preprocess tạo manifest train/val; mixed có split file explicit `data/split_vjepa_ac_car.json` | NN-JEPA rõ artifact hơn; JEPA tiện hơn trong train |
| Test split | Source train DROID chủ yếu theo train/eval config | Train AC chủ yếu train/val, không thấy test split riêng trong train loop AC | Hiện bỏ test độc lập, `test.jsonl` alias val để tương thích tool cũ | Cả hai repo RC đều chấp nhận được khi data ít; source không bắt buộc test riêng |
| Multi-root old servo | Không có khái niệm servo domain | Dùng `roots` trực tiếp trong config train | Build experiment riêng `servo_old_mix_v1` rồi train trên manifest/feature của experiment | JEPA gọn hơn; NN-JEPA dễ audit hơn |
| Domain servo | Không có domain token kiểu KDS/TowerPro | Thêm `domain_id` vào action token, action_dim=3 | Lưu `data_domain` để report, chưa đưa domain vào action/model | So với source thì NN-JEPA gần hơn vì không thêm domain; với mixed servo RC thì JEPA thực dụng hơn |
| DataLoader batch | Source dùng `DistributedSampler` + DataLoader batch thường | `SessionBatchSampler`, một batch cùng session | `DataLoader(... shuffle=True)`, batch có thể trộn nhiều session | NN-JEPA gần source hơn về sampler; JEPA tốt hơn cho memmap/cache I/O |

## 5. So sánh train V-JEPA 2.1 AC

| Phần | Source `vjepa2` public | JEPA của bạn bạn | NN-JEPA hiện tại | Cách nào chuẩn hơn |
|---|---|---|---|---|
| Mục tiêu train | Train action-conditioned predictor dự đoán latent token tương lai từ frozen target encoder | Train patch-token world model cho xe RC, phục vụ CEM closed-loop | Train AC predictor/world model từ feature cache, phục vụ offline eval/planning | Cả hai cùng hướng; source là mốc exact |
| Encoder trong train | `target_encoder(c)` chạy on-the-fly, `requires_grad=False`; từng frame được duplicate thành clip 2-frame trước khi encode | V-JEPA 2.1 ViT-L 384 frozen, encode offline | V-JEPA 2.1 ViT-B 384 frozen, encode offline | Source exact là on-the-fly. Offline cache vẫn đúng nếu encoder frozen và transform cố định |
| Encoder có backprop không | Không backprop target encoder trong loop AC public | Không, encode offline | Không, encode offline | Cả hai đúng tinh thần frozen encoder |
| Feature representation | Full token-level latent, không mean-pool; ViT-g latent dim 1408 trong source public DROID | Full patch tokens `(576,1024)` | Full patch tokens `(576,768)` | Cả hai đúng token-level; không tương thích source hoặc nhau |
| Predictor chính | `VisionTransformerPredictorAC` | `VJEPA2ACCar` | `SimpleACPredictor` hoặc `VJepaStyleACPredictor` official_lite | NN-JEPA `official_lite` gần source hơn ở kiến trúc mask/RoPE; JEPA gần hơn về system thực chiến |
| Config predictor | `pred_depth: 24`, `pred_embed_dim: 1024`, `pred_num_heads: 16`, `pred_is_frame_causal: true`; predictor map latent 1408 -> hidden 1024 -> latent 1408 | `pred_dim=512`, `depth=12`, `heads=8`, `dropout=0.1` | tiny hiện `pred_dim=128`, `depth=2`, `heads=4`, `dropout=0.0` | JEPA gần source hơn về scale; NN-JEPA tiny là bản thử nhanh |
| Action token | Source DROID dùng action 7D | `[steer, throttle, domain_id]`, throttle scale 6.67 | `[steering_cmd_t, throttle_cmd_t]`, normalize bằng train stats | Không bên nào exact vì xe RC khác robot DROID |
| State token | Source DROID dùng state 7D, extrinsics optional | `[speed,gx,gy,gz,ax,ay,az,rx,ry,rz,prev_steer,prev_throttle]` | `[yaw_rate_t,accel_x_t,accel_y_t,steering_last_t,throttle_last_t]` | JEPA giàu cảm biến hơn; NN-JEPA đơn giản hơn, ít nhiễu hơn |
| Temporal length | 8 frame thật, 7 transition teacher-forcing, rollout `auto_steps=2` | `horizon=4`, transition TF trên 3 cặp, rollout 2-step | `raw_frames_per_sample=8`, TF trên 7 cặp, rollout `auto_steps=2` | NN-JEPA gần source hơn về số frame/transition; JEPA nhẹ hơn nên train ổn hơn |
| Loss | Teacher forcing loss + autoregressive rollout loss, có `normalize_reps: true` | L1 teacher forcing + 2-step rollout | L1 teacher forcing + rollout loss | Cả hai gần mục tiêu; JEPA gần hơn nếu tính re-LN feedback |
| Rollout feedback | Source layer-norm target reps và prediction reps khi `normalize_reps` bật | Re-LN prediction trước khi feed back | Feature đã LN từ encoder; rollout hiện chưa re-LN `next_tokens` trong `compute_world_model_losses` | JEPA gần source hơn ở điểm này |
| Attention mask | `build_action_block_causal_attention_mask` trong `VisionTransformerPredictorAC` | Block-causal theo frame group | `simple`: time-causal mask; `official_lite`: action-block causal mask | NN-JEPA `official_lite` gần source nhất |
| RoPE/positional | `use_rope: true` trong config DROID | Learnable temporal/token pos, không RoPE trong `VJEPA2ACCar` | `official_lite` có RoPE attention custom | NN-JEPA `official_lite` gần source hơn |
| Drop path | Source class hỗ trợ `drop_path_rate`, config DROID không nhấn mạnh bật drop path | Không thấy drop path trong model car | Chưa có drop path trong official_lite hiện tại | Không phải khác biệt lớn hiện tại |
| Gradient checkpointing | `use_activation_checkpointing: true` | Có option `gradient_checkpointing: true` | Chưa có cho predictor | JEPA gần source hơn và tiết kiệm VRAM hơn |
| Precision train | `dtype: bfloat16` | BF16 autocast | Feature fp16 nhưng train loop chủ yếu float32 | JEPA gần source hơn |
| LR schedule | Config có `start_lr`, `lr`, `final_lr`, `warmup: 15`, train dài `epochs: 315` | Tự set LR mỗi step bằng cosine warmup | PyTorch `LambdaLR`, warmup rồi cosine decay | Cả hai gần tinh thần schedule; source train lâu hơn nhiều |
| Early stopping | Source config public không nhấn mạnh early stopping kiểu small data | Theo val loss, `patience=12`, `min_delta=1e-4` | Theo val loss, `patience=15`, warmup không tính | Đây là thêm cho RC small data, không phải điểm official |
| Checkpoint | Source save `encoder`, `predictor`, `target_encoder`, optimizer, scaler, epoch, lr | `last.pt`, `best.pt`, nhưng train AC không lưu optimizer/scheduler đầy đủ | `last_train.pt`, `last.pt`, `best.pt`, epoch checkpoints, lưu optimizer/scheduler/history | NN-JEPA gần source hơn về resume đầy đủ |
| W&B/logging | Source có logging cơ bản theo pipeline | Log train loss/tf/rollout/lr mỗi 50 step, val loss mỗi epoch | Log batch/epoch nhiều hơn, có grad/param stats optional | NN-JEPA tốt hơn cho debug experiment |
| Eval cuối train | Source tập trung loss train/val theo pipeline | Có `final_eval` rollout vs identity ratio | Có val mỗi epoch, optional test/eval/inference tools riêng | JEPA có metric rollout trực quan hơn; NN-JEPA rõ hơn về experiment |
| Closed-loop planner | Source AC train không phải planner RC hoàn chỉnh | Có CEM + graph + phone relay | Có offline CEM/planning tools, chưa full closed-loop như JEPA | JEPA vượt xa source train loop ở phần robot system |

## 6. Mức độ giống V-JEPA 2.1 AC official

| Tiêu chí | Source `vjepa2` public | JEPA của bạn bạn | NN-JEPA hiện tại | Cách nào chuẩn hơn |
|---|---|---|---|---|
| Frozen pretrained encoder | `target_encoder` frozen trong train loop | Có, encode offline | Có, encode offline | Cả hai đúng tinh thần frozen encoder |
| Full token-level latent, không mean-pool | Có | Có với `VJEPA2ACCar` | Có với feature cache hiện tại | Cả hai đúng |
| Action/state token conditioning | Có action/state token, extrinsics optional | Có action/state/domain token | Có action/state token | Cả hai đúng ý tưởng; NN-JEPA ít lệch hơn vì chưa thêm domain |
| Block-causal attention theo frame | Có `build_action_block_causal_attention_mask` | Có block-causal riêng | Có với `official_lite`; `simple` đơn giản hơn | NN-JEPA `official_lite` gần source hơn về mask |
| RoPE giống source Meta | `use_rope: true` | Không trong `VJEPA2ACCar`, dùng learnable temporal/token pos | Có trong `official_lite` custom | NN-JEPA `official_lite` gần source hơn |
| Predictor depth/scale | Public DROID config depth 24, heads 16, predictor hidden dim 1024; encoder latent dim 1408 | Depth 12, heads 8, pred dim 512 | Tiny depth 2, heads 4, pred dim 128 | JEPA gần source hơn về scale |
| Input resolution public DROID | 256px | 384px | 384px, nhưng feature extractor cần tránh ảnh processed 224 | Source exact là 256; nếu theo V-JEPA 2.1 384 thì JEPA/NN-JEPA đều hợp lý |
| Temporal sampling | `dataset_fpcs=8`, `fps=4`; DataLoader dùng `tubelet_size=1`, còn target encoder duplicate frame để xử lý `tubelet_size=2` | `horizon=4`, `frame_stride=2` | `raw_frames_per_sample=8`, `frame_stride=2` hoặc `target_fps` | NN-JEPA gần source hơn về T=8; NN-JEPA `target_fps` gần source nhất nếu bật |
| State/action giống robot official | DROID 7D action/state | Đổi sang xe RC 12D state, 3D action | Đổi sang xe RC 5D state, 2D action | Không bên nào exact; JEPA giàu thông tin hơn cho RC |
| Train encoder tiếp | Không train encoder trong AC loop public | Không | Không | Cả hai đúng |
| Precision/VRAM | BF16 + activation checkpointing | BF16 + checkpointing | Float32 train path, chưa checkpointing | JEPA gần source hơn |
| Checkpoint resume | Source save optimizer/scaler | Chưa đầy đủ optimizer/scheduler trong train AC | Có optimizer/scheduler/global step | NN-JEPA gần source hơn về resume |
| Robot-ready closed-loop | Source public AC không phải stack RC hoàn chỉnh | Có CEM + dynamics + graph + phone relay | Có offline CEM/planning tools, chưa full closed-loop như JEPA | JEPA mạnh nhất về triển khai xe thật |

Kết luận: nếu chấm “giống source `vjepa2` public” theo kiến trúc predictor, NN-JEPA `official_lite` gần hơn ở action-block causal mask và RoPE. Nếu chấm theo khả năng train ổn trên xe RC và triển khai robot, JEPA của bạn bạn mạnh hơn rõ ràng vì có sync, feature raw-frame, domain token, BF16, gradient checkpointing, CEM, graph và closed-loop stack. Nếu chấm theo scale model, JEPA gần source hơn NN-JEPA tiny; nếu chấm theo resume/audit experiment, NN-JEPA tốt hơn.

## 7. Những khác biệt có ảnh hưởng lớn tới tốc độ và OOM

| Nguyên nhân | JEPA | NN-JEPA | Ảnh hưởng |
|---|---|---|---|
| Sequence length trong predictor | `4 * (576 + 2) = 2312 token` | teacher-forcing input khoảng `7 * (576 + 2) = 4046 token` với sample 8 frame | Attention của NN-JEPA nặng hơn nhiều |
| Batch cache | Batch cùng session | Batch shuffle global | NN-JEPA dễ nhảy file nhiều hơn |
| Feature dim | 1024 | 768 | JEPA mỗi token nặng hơn |
| Precision | BF16 autocast | Float32 train path | NN-JEPA dễ tốn VRAM hơn dù dim nhỏ hơn |
| Gradient checkpointing | Có | Chưa có | JEPA train được model sâu hơn trên 16 GB |
| Prefetch | JEPA DataLoader đơn giản, không persistent/prefetch_factor custom trong AC train | NN-JEPA dùng `persistent_workers` và `prefetch_factor` từ settings | NN-JEPA có thể tốn RAM hơn khi batch lớn |
| Eval val | JEPA dùng cùng sampler theo session và batch_size train | NN-JEPA dùng eval_batch_size riêng, từng gặp OOM eval do transformer fastpath | NN-JEPA đã có disable eval fastpath cho predictor |

## 8. Những điểm NN-JEPA đang làm tốt hơn JEPA

| Điểm | Vì sao tốt |
|---|---|
| Feature metadata rõ | `metadata.json` ghi encoder, checkpoint, dtype, token/frame, embed dim, split, frame count |
| Cache validation tốt | Extractor có thể skip session compatible, báo incompatible, seed symlink từ feature dir khác |
| Hydra config | Dễ chạy experiment bằng `experiment=...`, override tham số sạch |
| Resume train | Lưu optimizer, scheduler, global step, phase `train_complete_waiting_val`, có thể resume sau cúp điện tốt hơn |
| W&B logging | Có batch/epoch metrics, grad stats, param stats, run id/resume |
| Manifest explicit | Train/val/test alias rõ, sample có state/action/meta/source path |
| Preprocess report | Có report dropped/missing/outlier/action range, hữu ích debug data bẩn |

## 9. Những điểm JEPA đang làm tốt hơn NN-JEPA

| Điểm | Vì sao tốt |
|---|---|
| Robot-ready | Có route web, graph navigation, phone relay, CEM closed-loop |
| Data sync đúng hơn cho app JEPA | Nội suy action/IMU theo scene time và camera delay rõ ràng |
| Feature extraction đúng input hơn | Đọc raw frame rồi resize trực tiếp tới encoder size |
| Session-batch sampler | Giảm nhảy file, giảm áp lực RAM/I/O với feature cache lớn |
| Domain token | Model biết KDS/TowerPro khác calibration |
| State giàu hơn | Dùng speed, full gyro, full accel, rotvec, prev action |
| BF16 + checkpointing | Train được predictor sâu hơn trên GPU 16 GB |
| CEM thực dụng hơn | Có dynamics, prev-action, domain, warm-start policy prior |
| Frozen split cạnh checkpoint | Tránh val set thay đổi khi thêm session mới |

## 10. Rủi ro nếu dùng lẫn JEPA và NN-JEPA

Không được dùng lẫn checkpoint hoặc feature cache giữa hai bên nếu không convert kỹ.

| Artifact | JEPA | NN-JEPA | Có dùng lẫn được không |
|---|---|---|---|
| Feature patch 384 | `(N,576,1024)`, ViT-L | `(N,576,768)`, ViT-B hiện tại | Không |
| Predictor checkpoint | `VJEPA2ACCar`, key `model`, state/action 12D/3D | `predictor_state_dict`, state/action 5D/2D | Không |
| Split | `split.json` cạnh checkpoint | Manifest jsonl hoặc split file explicit | Chỉ tham khảo session list, không dùng trực tiếp nếu tên/domain khác |
| State stats | Lưu `state_mean`, `state_std` trong checkpoint JEPA | Lưu normalization metadata trong checkpoint NN-JEPA | Không dùng chéo |
| CEM planner | Expect model `rollout(z0, states, actions)` với token shape `(B,T,N,D)` | NN-JEPA planner load runtime riêng | Ý tưởng port được, object không dùng trực tiếp |

## 11. Việc nên port từ JEPA sang NN-JEPA

Ưu tiên 1: sửa feature extractor NN-JEPA dùng raw frame.

Hiện NN-JEPA `FrameFeatureDataset.__getitem__` đọc `sample["frame_path"]`. Trường này đang là ảnh processed 224. Nên đổi logic:

```python
image_path = sample.get("source_frame_path") or sample["frame_path"]
```

Như vậy encoder V-JEPA 2.1 sẽ nhận raw frame được resize trực tiếp lên 384, gần JEPA hơn.

Ưu tiên 2: thêm `SessionBatchSampler` cho feature dataset.

Mục tiêu:

| Lợi ích | Giải thích |
|---|---|
| Giữ memmap cache nóng | Một batch cùng session tránh đọc nhiều `.npy` lớn cùng lúc |
| Giảm RAM/I/O | Ít worker cùng lúc nhảy nhiều file |
| Ổn định train/val | Có thể giảm GPU rảnh do data loading |

Ưu tiên 3: thêm domain token cho mixed servo.

NN-JEPA hiện chỉ lưu `data_domain`, chưa đưa vào model. Nếu train mix old servo + current servo, nên thêm một cột action/state domain để predictor biết servo calibration khác nhau.

Ưu tiên 4: thêm BF16 autocast cho train predictor.

JEPA đang dùng:

```python
with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    loss, tf, ro = _losses(...)
```

NN-JEPA có thể thêm option `--amp-dtype bf16/fp32` để vừa an toàn vừa tiết kiệm VRAM.

Ưu tiên 5: thêm gradient checkpointing cho `official_lite`.

Nếu muốn train official_lite small/base trên 16 GB VRAM, gradient checkpointing quan trọng hơn tăng batch.

Ưu tiên 6: tạo experiment NN-JEPA dùng ViT-L 384.

Chỉ nên làm sau khi pipeline ViT-B chạy ổn. Nếu dùng ViT-L, phải extract lại feature và train predictor lại từ đầu vì `embed_dim` đổi từ 768 sang 1024.

## 12. Có nên chuyển NN-JEPA sang giống JEPA ngay không?

Không nên chuyển toàn bộ ngay. Nên port từng phần, vì JEPA là hệ thống robot thực chiến đã phức tạp, còn NN-JEPA hiện đang dùng để thử nghiệm mô hình có kiểm soát.

Thứ tự khuyến nghị:

| Bước | Làm gì | Lý do |
|---|---|---|
| 1 | Sửa extractor dùng `source_frame_path` | Đây là bug/rủi ro chất lượng rõ nhất |
| 2 | Chạy lại extract feature ViT-B fp16 | Feature cũ có thể đang từ ảnh 224 |
| 3 | Train lại official_lite tiny mix | Kiểm tra loss/val/W&B ổn |
| 4 | Thêm session-batch sampler | Giảm I/O/RAM, không đổi bản chất sample |
| 5 | Thêm domain token | Quan trọng cho mixed KDS/TowerPro |
| 6 | Thêm BF16 + gradient checkpointing | Mở đường cho official_lite small/base |
| 7 | So sánh ViT-B vs ViT-L | Chỉ khi pipeline đã ổn, vì ViT-L tốn disk/VRAM hơn |
| 8 | Port CEM/policy prior kiểu JEPA | Sau khi world model đủ tốt |

## 13. Kết luận cuối

JEPA của bạn bạn hiện là nhánh mạnh về triển khai robot thật: train model sâu hơn, feature ViT-L, domain token, batch theo session, graph navigation, CEM và web route planner.

NN-JEPA hiện là nhánh mạnh về quản lý experiment: manifest rõ, Hydra, W&B, resume, feature metadata, audit dễ hơn.

Nếu mục tiêu ngắn hạn là train ổn định qua đêm và phân tích loss, giữ NN-JEPA và port từng cải tiến từ JEPA là hợp lý nhất.

Nếu mục tiêu là chạy xe thật closed-loop sớm, JEPA hiện đã có nhiều mảnh ghép hơn. Nhưng không nên trộn checkpoint/feature giữa hai repo vì encoder dim, state/action dim và cache layout khác nhau.

Việc nên làm ngay trong NN-JEPA: sửa extractor dùng raw `source_frame_path`, rồi extract lại feature fp16 cho experiment cần train.
