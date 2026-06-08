# Cập nhật JEPA 2026-06-08 và ghi chú về Hydra

File này ghi lại các cập nhật mới trong repo `JEPA/` sau khi pull, đồng thời giải thích nhanh Hydra là gì và vì sao nó phù hợp cho việc cấu hình train bằng file YAML.

## 1. Trạng thái repo JEPA sau pull

Repo `JEPA/` hiện ở commit:

```text
be50793 Add visual navigation (TopoGraph subgoal) + control on current servo
```

Các file chính được thêm hoặc cập nhật:

```text
JEPA/src/jepa_wm/nav/graph.py
JEPA/src/jepa_wm/nav/__init__.py
JEPA/src/jepa_wm/planning/cem.py
JEPA/scripts/build_graph.py
JEPA/scripts/eval_navigation.py
JEPA/scripts/eval_goal_reaching.py
JEPA/scripts/viz_route.py
JEPA/configs/train/vjepa_ac_towerpro.yaml
JEPA/configs/train/vjepa_ac_mixed.yaml
JEPA/robot/tools/pull_drive.py
JEPA/scripts/eval_lewm.py
JEPA/src/jepa_wm/data/dataset.py
JEPA/src/jepa_wm/engine/train.py
JEPA/docs/HANDOFF.md
```

Điểm quan trọng: đây là cập nhật của repo `JEPA/`, tức repo phần cứng, recorder, sync, navigation và thí nghiệm riêng bên đó. Code train chính trong repo `NN-JEPA` chưa tự thay đổi theo các cập nhật này.

## 2. Ý tưởng lớn mới: navigation 2 tầng

Trong `JEPA/docs/HANDOFF.md`, hướng mới được mô tả là:

```text
tự lái = visual goal-reaching cục bộ + topological graph ảnh subgoal
```

Nói đơn giản:

```text
ảnh hiện tại -> định vị trong graph -> chọn chuỗi ảnh subgoal -> CEM chọn action để đi tới từng subgoal
```

Hệ này được tách thành 2 tầng:

```text
Tầng 1: Navigation
  action-agnostic
  chỉ dùng ảnh/V-JEPA latent + GPS
  nhiệm vụ: biết xe đang ở đâu và route tới goal qua các subgoal

Tầng 2: Control
  servo-specific
  dùng world model + CEM
  nhiệm vụ: chọn steering/throttle để tới subgoal gần
```

Việc tách như vậy hợp lý vì:

- navigation cần nhận ra địa điểm, không cần biết servo nào đang gắn trên xe
- control phụ thuộc vào servo, throttle, response xe thật
- nếu đổi servo, tầng navigation có thể giữ lại, còn tầng control cần train/eval lại

## 3. TopoGraph mới trong JEPA

File chính:

```text
JEPA/src/jepa_wm/nav/graph.py
```

`TopoGraph` là graph topological dùng cho visual navigation.

Mỗi node là một frame đã ghi:

```text
node = {
  latent V-JEPA của frame,
  GPS local mét,
  lat/lon,
  heading,
  session_id,
  frame_idx
}
```

Có 2 loại edge:

```text
temporal edge:
  nối các frame liên tiếp trong cùng session
  ý nghĩa: người đã lái qua đoạn này, nên coi là đường có thể đi

loop-closure edge:
  nối các frame giống nhau giữa các session khác nhau
  dùng cosine similarity trên latent V-JEPA
  có GPS gate để tránh nhầm hai chỗ nhìn giống nhau nhưng ở xa nhau
```

Các hàm quan trọng:

```text
localize()
  tìm node gần nhất với latent ảnh hiện tại
  nếu có GPS prior thì giới hạn tìm trong vùng gần GPS

plan_route()
  chạy Dijkstra trên graph để tìm route từ node hiện tại tới node goal

extract_subgoals()
  biến route dài thành danh sách node subgoal cách nhau vài mét
```

Ý nghĩa với xe RC: nếu goal ở xa hoặc không nhìn thấy trực tiếp, graph sẽ chia goal xa thành nhiều ảnh subgoal gần hơn. Sau đó controller chỉ cần giải bài toán cục bộ: đi từ ảnh hiện tại tới ảnh subgoal gần.

## 4. Script graph mới

### 4.1. Build graph

File:

```text
JEPA/scripts/build_graph.py
```

Nhiệm vụ:

- đọc latent đã encode
- đọc raw session và GPS
- tạo node
- tạo temporal edge
- tạo loop-closure edge
- lưu graph ra `.pt`

Ví dụ trong JEPA:

```bash
PYTHONPATH=src python scripts/build_graph.py \
  --root data/latents:data/raw:kds \
  --root data/latents_towerpro:data/raw_towerpro:towerpro \
  --out data/graph/topograph.pt
```

### 4.2. Eval navigation

File:

```text
JEPA/scripts/eval_navigation.py
```

Nó đánh giá:

```text
localization:
  ảnh này nằm gần node nào trong graph

routing:
  random start/goal có route được không

route-vs-actual:
  route được plan có bám theo hành lang người từng lái không
```

### 4.3. Visualize route

File:

```text
JEPA/scripts/viz_route.py
```

Nhiệm vụ: vẽ route ra ảnh PNG để kiểm tra trực quan.

## 5. CEMPlannerLatent mới

File:

```text
JEPA/src/jepa_wm/planning/cem.py
```

Trong file này có thêm class:

```text
CEMPlannerLatent
```

Nó dùng cho model dạng latent world model, tức model có thể:

```text
s0 + chuỗi action -> rollout latent tương lai
```

Cách CEM hoạt động:

```text
1. Sample nhiều chuỗi action ngẫu nhiên
2. Cho world model rollout từng chuỗi action
3. Tính điểm: latent cuối có gần goal latent không
4. Chọn top action sequence tốt nhất
5. Refit phân phối sample quanh nhóm tốt nhất
6. Lặp vài lần
7. Trả về action đầu tiên của sequence tốt nhất
```

Với xe RC:

```text
input:
  latent hiện tại
  latent goal/subgoal

output:
  steering, throttle cho bước tiếp theo
```

Điểm này quan trọng vì model hiện tại của mình cũng đang là world model trong latent space. Nếu sau này muốn xe tự chọn action, không chỉ train loss, thì CEM là hướng có thể port sang NN-JEPA.

## 6. Eval goal-reaching mới

File:

```text
JEPA/scripts/eval_goal_reaching.py
```

Script này đánh giá controller offline.

Nó lấy một đoạn trong validation:

```text
s_t là latent hiện tại
s_{t+d} là goal latent
```

Sau đó hỏi CEM:

```text
hãy chọn action sequence để từ s_t đi tới s_{t+d}
```

Metric chính:

```text
CEM:
  lỗi latent cuối của CEM so với goal

teacher:
  lỗi khi dùng action thật của người lái

random:
  lỗi khi dùng action random

CEM/random:
  < 1 nghĩa là CEM tốt hơn random

delta steer / delta throttle:
  action CEM chọn lệch bao nhiêu so với action thật của người lái
```

Đây là bài test rất có ích vì nó kiểm tra model có đủ tốt để planning không, thay vì chỉ nhìn train/val loss.

## 7. Config train mới cho TowerPro và mixed data

File:

```text
JEPA/configs/train/vjepa_ac_towerpro.yaml
JEPA/configs/train/vjepa_ac_mixed.yaml
```

Ý nghĩa:

```text
vjepa_ac_towerpro.yaml:
  train riêng trên data servo TowerPro hiện tại

vjepa_ac_mixed.yaml:
  train trên data mixed KDS + TowerPro
```

Điểm đáng chú ý nhất:

```yaml
action_scale: [1.0, 6.67]
```

Lý do:

```text
steering thường nằm gần [-1, 1]
throttle thực tế chỉ khoảng [-0.15, 0.15]
```

Nếu đưa raw action vào model, throttle nhỏ hơn steering khoảng 6.67 lần, nên model dễ coi nhẹ throttle.

Scale `[1.0, 6.67]` giúp:

```text
steering giữ nguyên
throttle được scale lên gần cùng biên độ với steering
```

Đây là điểm NN-JEPA nên cân nhắc port nếu thấy model hiện tại học throttle yếu.

## 8. Dataset và train engine bên JEPA thay đổi gì

File:

```text
JEPA/src/jepa_wm/data/dataset.py
JEPA/src/jepa_wm/engine/train.py
```

Thay đổi chính:

```text
LatentTransitionDataset thêm action_scale
train.py truyền action_scale từ YAML config vào dataset
```

Nghĩa là scaling action được cấu hình bằng YAML, không hard-code trong model.

## 9. pull_drive.py được cập nhật

File:

```text
JEPA/robot/tools/pull_drive.py
```

Điểm mới:

```bash
--dest data/raw_towerpro
```

Trước đây tool chủ yếu kéo data về `data/raw`. Giờ có thể chỉ định thư mục đích, ví dụ kéo data servo mới về `data/raw_towerpro`.

Tool cũng xử lý lỗi unzip tốt hơn:

```text
nếu unzip lỗi -> xóa folder extract lỗi -> bỏ qua zip đó
```

## 10. eval_lewm.py được cập nhật

File:

```text
JEPA/scripts/eval_lewm.py
```

Điểm mới:

```bash
--domains kds680hv towerpro
--device cuda
```

Ý nghĩa:

- có thể eval model trên một domain cụ thể
- ví dụ train mixed nhưng chỉ eval held-out TowerPro
- tránh đánh giá bị lẫn domain và hiểu sai model có transfer tốt không

## 11. Phát hiện quan trọng: single-frame latent có thể thiếu motion

Trong `JEPA/docs/HANDOFF.md`, có một cảnh báo quan trọng:

```text
V-JEPA 2.1 là video model
nhưng hiện đang feed T=1 frame
```

Với `tubelet_size = 2`:

```text
T=1:
  vẫn ra latent hợp lệ
  nhưng gần như là single-frame representation
  không có motion/tốc độ

T=2:
  có 1 tubelet temporal
  bắt đầu có thông tin motion

T=4 hoặc T=8:
  có nhiều thông tin chuyển động hơn
```

Kết luận bên JEPA:

```text
single-frame tốt cho navigation/place recognition
multi-frame clip tốt hơn cho control
```

Vì sao:

- navigation cần nhận ra nơi chốn, motion có thể gây nhiễu
- control cần biết xe đang tiến nhanh/chậm, đang đổi hướng thế nào
- throttle đặc biệt cần thông tin motion/tốc độ

Đây là điểm rất đáng lưu ý cho NN-JEPA. Model hiện tại của NN-JEPA đang train trên chuỗi nhiều frame, nhưng feature cache từng frame vẫn được encode như từng frame riêng. Nếu sau này thấy throttle yếu, hướng nâng cấp nên là encode clip multi-frame cho control.

## 12. Ảnh hưởng hiện tại tới NN-JEPA

Hiện tại:

```text
NN-JEPA chưa dùng TopoGraph
NN-JEPA chưa dùng CEMPlannerLatent từ JEPA
NN-JEPA chưa dùng YAML config kiểu JEPA
NN-JEPA chưa dùng action_scale [1.0, 6.67]
NN-JEPA vẫn train predictor riêng trong src/
```

Nhưng các cập nhật từ JEPA gợi ý roadmap rất rõ:

```text
Bước 1:
  giữ pipeline train hiện tại
  train predictor tiny/base cho ổn

Bước 2:
  thêm eval goal-reaching tương tự JEPA để kiểm tra model có planning được không

Bước 3:
  port CEMPlannerLatent sang NN-JEPA

Bước 4:
  nếu cần tự lái goal xa, port TopoGraph để chọn subgoal

Bước 5:
  thử action_scale hoặc action normalizer tốt hơn cho throttle

Bước 6:
  nghiên cứu encode multi-frame clip cho control
```

## 13. Hydra là gì?

Hydra là một framework cấu hình cho Python, thường dùng trong ML/research để quản lý config bằng YAML.

Nói đơn giản: đúng, Hydra giúp mình không cần nhớ một lệnh train dài đầy tham số.

Thay vì chạy:

```bash
PYTHONPATH=src python3 -m tools.train_rc_jepa_ac_features \
  --features-dir data/processed/features/vjepa2_1_vitb_384_ema_fp32 \
  --manifest-dir data/processed/manifests \
  --output-dir checkpoints/rc_jepa_ac_vitb_features_20260607_tiny \
  --model-size tiny \
  --epochs 20 \
  --batch-size 10 \
  --eval-batch-size 2 \
  --num-workers 8 \
  --lr 1e-4 \
  --warmup-epochs 2 \
  --early-stopping-patience 5 \
  --wandb-project nn-jepa-rc \
  --wandb-tags tiny
```

Ta có thể cấu hình bằng YAML rồi chạy ngắn hơn, ví dụ:

```bash
PYTHONPATH=src python3 -m tools.train_hydra experiment=rc_jepa_tiny
```

Hoặc override vài tham số ngay trên CLI:

```bash
PYTHONPATH=src python3 -m tools.train_hydra experiment=rc_jepa_tiny train.epochs=5 model.size=small
```

## 14. Hydra không chỉ là đọc YAML

Hydra thường mạnh hơn một YAML parser thường ở các điểm:

```text
config composition:
  ghép nhiều file YAML nhỏ lại thành một config lớn

CLI override:
  sửa một giá trị config ngay trên command line

multirun/sweep:
  chạy nhiều tổ hợp hyperparameter

structured output:
  tự tạo thư mục run/log/config cho từng lần chạy

OmegaConf:
  hỗ trợ biến nội suy như ${data.root}, ${now:%Y%m%d}
```

Ví dụ cấu trúc config:

```text
configs/
  config.yaml
  data/
    rc_features.yaml
  model/
    predictor_tiny.yaml
    predictor_base.yaml
  train/
    default.yaml
  experiment/
    rc_jepa_tiny.yaml
    rc_jepa_base.yaml
```

Ví dụ `experiment/rc_jepa_tiny.yaml`:

```yaml
defaults:
  - /data: rc_features
  - /model: predictor_tiny
  - /train: default

output_dir: checkpoints/rc_jepa_ac_vitb_features_20260607_tiny

train:
  epochs: 20
  batch_size: 10
  eval_batch_size: 2
  lr: 1.0e-4
  warmup_epochs: 2
  early_stopping_patience: 5

wandb:
  project: nn-jepa-rc
  tags: [tiny]
```

Ví dụ `model/predictor_tiny.yaml`:

```yaml
size: tiny
predictor_dim: 128
predictor_depth: 2
predictor_heads: 4
dropout: 0.0
```

## 15. Hydra có hợp với NN-JEPA không?

Có, vì NN-JEPA hiện đã có nhiều tham số:

```text
data/features-dir
manifest-dir
model-size tiny/small/base
batch-size
eval-batch-size
epochs
lr
warmup
early stopping
wandb project/entity/tags
output-dir
resume-from
```

Nếu cứ dùng CLI dài, rất dễ:

- quên tham số
- chạy nhầm output dir
- train tiny nhưng ghi đè base
- resume nhầm checkpoint
- W&B log sang entity/project không mong muốn

Hydra giúp gom các cấu hình đó thành experiment rõ ràng:

```text
rc_jepa_tiny_debug
rc_jepa_base_full
rc_jepa_small_sweep_lr
rc_jepa_base_resume
```

## 16. Nhưng có một vài điểm phải cẩn thận với Hydra

Hydra có một behavior dễ gây nhầm:

```text
mặc định Hydra có thể đổi working directory sang thư mục run mới
```

Điều này có thể làm đường dẫn tương đối như:

```text
data/processed/manifests
checkpoints/...
```

bị lệch nếu không cấu hình kỹ.

Khi dùng Hydra cho project này, nên đặt:

```yaml
hydra:
  job:
    chdir: false
```

Hoặc dùng đường dẫn tuyệt đối dựa trên repo root.

Ngoài ra, Hydra rất mạnh nhưng cũng có thể làm project rối nếu config tree quá phức tạp. Với NN-JEPA, nên bắt đầu đơn giản:

```text
1 file experiment YAML cho tiny
1 file experiment YAML cho base
1 train wrapper đọc Hydra config rồi gọi lại hàm train hiện tại
```

Không nên refactor toàn bộ train loop ngay từ đầu.

## 17. Đề xuất thực tế cho NN-JEPA

Thứ tự hợp lý:

```text
1. Giữ CLI hiện tại để không phá pipeline đang chạy.
2. Thêm một wrapper Hydra riêng, ví dụ tools/train_rc_jepa_ac_features_hydra.py.
3. Tạo config tiny/base bằng YAML.
4. Wrapper chuyển YAML config thành argparse.Namespace rồi gọi lại logic train hiện tại.
5. Sau khi chạy ổn, mới cân nhắc bỏ bớt lệnh CLI dài.
```

Tức là Hydra nên là lớp config bên ngoài, không nên ép sửa toàn bộ model/dataset ngay.

## 18. Tóm tắt ngắn

Repo `JEPA/` mới thêm:

```text
TopoGraph:
  chọn route và subgoal bằng ảnh + GPS

CEMPlannerLatent:
  chọn action để đi tới latent goal/subgoal

eval_goal_reaching:
  kiểm tra offline xem CEM có điều khiển tốt không

config TowerPro/mixed:
  thêm action_scale để throttle không bị model coi nhẹ

cảnh báo single-frame:
  control có thể cần multi-frame latent để học motion/throttle tốt hơn
```

Hydra:

```text
đúng là công cụ giúp quản lý train config bằng YAML
giúp lệnh train ngắn hơn, rõ hơn, ít nhầm hơn
phù hợp với NN-JEPA, nhưng nên thêm từ từ bằng wrapper riêng
```
