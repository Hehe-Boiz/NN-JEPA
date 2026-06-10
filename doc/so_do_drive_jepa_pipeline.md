# Sơ Đồ Hoạt Động Của Drive-JEPA

Ngày cập nhật: 2026-06-10

Mục tiêu của file này là làm rõ:

```text
1. V-JEPA trong Drive-JEPA dùng để làm gì
2. Predictor của V-JEPA nằm ở đâu
3. Planner của Drive-JEPA nằm ở đâu
4. Perception-free và perception-based khác nhau thế nào
```

---

## 1. Bức Tranh Tổng Thể

Drive-JEPA có thể hiểu là:

```text
Tầng 1: học representation bằng V-JEPA
Tầng 2: dùng representation đó cho planning
Tầng 3: chọn trajectory tốt nhất
```

Sơ đồ lớn:

```text
Driving videos
    |
    v
V-JEPA pretraining
    |
    v
Pretrained ViT encoder
    |
    +------------------------------+
    |                              |
    v                              v
Perception-free planner        Perception-based planner
    |                              |
    v                              v
1 trajectory                   many proposals + scorer
    |                              |
    +--------------+---------------+
                   |
                   v
            final trajectory
```

Điểm quan trọng:

```text
V-JEPA predictor không phải planner cuối.
Planner cuối là module khác, chỉ dùng feature từ encoder V-JEPA.
```

---

## 2. Tầng 1: V-JEPA Pretraining

Đây là phần self-supervised.

Mục tiêu:

```text
Học encoder ViT để sinh latent representation tốt cho video lái xe.
```

Sơ đồ:

```text
Video clip
    |
    +------------------------------+
    |                              |
    v                              v
Context view                    Target view
(bị che một phần)               (phần cần dự đoán)
    |                              |
    v                              v
Encoder E_theta                EMA encoder E_theta_bar
    |                              |
    v                              v
Context latent                 Target latent
    |                              |
    v                              |
Predictor P_phi -------------------+
    |
    v
Predicted target latent
    |
    v
L1 loss với target latent
```

Viết ngắn theo công thức:

```text
video -> encoder -> latent
latent context -> predictor -> predicted future latent
predicted future latent so với EMA target latent
```

Điểm cần nhớ:

```text
Encoder: lấy latent biểu diễn từ image/video
Predictor: dự đoán latent target/future từ latent context
```

Cho nên nếu hỏi:

```text
"V-JEPA trong Drive-JEPA có phải chỉ để lấy latent biểu diễn không?"
```

thì câu trả lời chính xác là:

```text
Không hoàn toàn.

Trong giai đoạn pretraining:
  encoder lấy latent
  predictor học dự đoán latent tương lai / latent bị mask

Trong giai đoạn planner downstream:
  chủ yếu dùng encoder đã pretrain để lấy feature
  predictor JEPA không còn là planner chính
```

---

## 3. V-JEPA Predictor Là Gì

Đây là predictor kiểu JEPA, không phải trajectory head.

Sơ đồ chi tiết:

```text
context image/video tokens
    |
    v
encoder latent tokens x
    |
    v
project sang predictor dim
    |
    v
ghép thêm action token + state token
    |
    v
Transformer AC blocks
    |
    v
bỏ action/state tokens
    |
    v
project về latent dim gốc
    |
    v
predicted latent tokens
```

Nó output:

```text
latent tokens
```

Nó không output:

```text
steering
throttle
waypoint trajectory
```

Nghĩa là:

```text
V-JEPA predictor học dynamics trong latent space.
Nó không phải head trực tiếp điều khiển xe.
```

---

## 4. Tầng 2A: Drive-JEPA Perception-Free

Đây là bản nhẹ hơn.

Input:

```text
2 front-view images
ego status
```

Sơ đồ:

```text
2 front images
    |
    v
Pretrained V-JEPA encoder
    |
    v
Spatiotemporal image features
    |
    +------------------------+
    |                        |
    v                        v
image feature tokens      ego status
    |                        |
    +-----------concat--------+
                |
                v
Transformer decoder với learnable waypoint queries
                |
                v
MLP head
                |
                v
future waypoints
```

Output:

```text
1 trajectory
= chuỗi waypoint tương lai
```

Sơ đồ shape trực quan:

```text
images -> V-JEPA encoder -> feature tokens F_t
status -> linear -> status token
[F_t, status token] -> transformer decoder
learnable queries Q -> attend vào F_t
decoder output -> MLP -> [x, y, heading] x M
```

Điểm chính:

```text
Ở nhánh này, V-JEPA encoder chỉ đóng vai trò feature extractor.
Trajectory được sinh bởi transformer decoder riêng.
```

---

## 5. Tầng 2B: Drive-JEPA Perception-Based

Đây là bản mạnh hơn trong paper.

Input:

```text
2 front-view images
ego status
BEV/map/meta feature theo NAVSIM
```

Sơ đồ tổng:

```text
2 front images
    |
    v
Pretrained / fine-tuned V-JEPA image encoder
    |
    v
Image features

ego status
    |
    v
ego feature

learnable proposal embeddings
    |
    v
initial proposal queries
    |
    +------------------------------+
    | iterative refinement x L     |
    |                              |
    v                              |
decode current proposals           |
    |                              |
    v                              |
waypoint proposals                 |
    |                              |
    v                              |
BEV / deformable attention <-------+
    |
    v
refined proposal features
```

Sau vòng refine cuối:

```text
proposal features
    |
    v
scorer
    |
    v
score cho từng proposal
    |
    v
chọn proposal tốt nhất
    |
    v
final trajectory
```

---

## 6. Proposal-Centric Planner Hoạt Động Ra Sao

Đây là phần dễ nhầm nhất.

Thay vì:

```text
predict đúng 1 trajectory
```

Drive-JEPA làm:

```text
predict nhiều candidate trajectories cùng lúc
```

Ví dụ:

```text
proposal_num = 32
num_poses = 8
```

thì output proposal có thể hình dung là:

```text
32 trajectory proposals
mỗi proposal có 8 waypoint
mỗi waypoint là (x, y, heading)
```

Sơ đồ:

```text
proposal feature
    |
    v
MLP
    |
    v
proposal[1] = 8 waypoint
proposal[2] = 8 waypoint
...
proposal[32] = 8 waypoint
```

Sau đó dùng image feature để refine lại các proposal này.

Ý nghĩa:

```text
Model không bị ép chỉ sinh ra một đáp án duy nhất ngay từ đầu.
Nó được phép duy trì nhiều khả năng lái khác nhau.
```

Đây là nền của multimodal behavior.

---

## 7. Multimodal Trajectory Distillation Nằm Ở Đâu

Paper nói:

```text
Mỗi scene thường chỉ có 1 human trajectory.
Nhưng trong thực tế có thể có nhiều trajectory hợp lệ.
```

Nên Drive-JEPA thêm pseudo-teacher trajectories từ simulator.

Sơ đồ:

```text
Training scene
    |
    +--------------------------+
    |                          |
    v                          v
human trajectory          trajectory vocabulary
                               |
                               v
                       simulator scoring
                               |
                               v
                  pseudo-teacher trajectories
```

Khi train proposals:

```text
proposal set
    |
    +-----------------------------+
    |                             |
    v                             v
gần human trajectory         gần pseudo-teacher trajectories
    |                             |
    +-------------loss------------+
```

Ý nghĩa:

```text
Drive-JEPA không chỉ học bắt chước 1 quỹ đạo người lái.
Nó học phân bố proposal đa dạng hơn.
```

---

## 8. Scorer Là Gì

Sau khi có nhiều proposals, phải chọn cái nào tốt nhất.

Sơ đồ:

```text
proposal features
    |
    v
pool theo waypoint
    |
    v
MLP scorer
    |
    v
score cho từng proposal
```

Scorer học các tín hiệu như:

```text
collision risk
route/area compliance
comfort
progress
```

Cuối cùng:

```text
proposal có score cao nhất sẽ được chọn làm trajectory cuối
```

---

## 9. Momentum-Aware Selection Là Gì

Vấn đề:

```text
Nếu chỉ chọn proposal có score cao nhất ở từng frame độc lập,
trajectory có thể bị giật giữa frame t-1 và t.
```

Drive-JEPA sửa bằng cách:

```text
thêm comfort / smoothness term
so proposal hiện tại với trajectory đã chọn ở frame trước
```

Sơ đồ:

```text
selected trajectory at t-1
             |
             v
 so sánh với proposals ở t
             |
             v
 distortion / comfort score
             |
             v
 recalibrate proposal score
             |
             v
 chọn proposal mượt hơn
```

Nghĩa là:

```text
Không chỉ "đúng" theo score.
Mà còn "mượt" theo thời gian.
```

---

## 10. Khác Nhau Giữa 3 Thứ

### 10.1. V-JEPA Encoder

Vai trò:

```text
lấy latent representation từ image/video
```

Sơ đồ:

```text
image/video -> encoder -> latent tokens
```

### 10.2. V-JEPA Predictor

Vai trò:

```text
dự đoán latent target/future từ latent context
```

Sơ đồ:

```text
latent context -> predictor -> predicted latent target
```

### 10.3. Drive-JEPA Planner

Vai trò:

```text
dùng feature từ encoder để sinh trajectory
```

Sơ đồ:

```text
V-JEPA features + ego status -> planner -> trajectory/proposals
```

Tóm lại:

```text
Encoder = lấy biểu diễn
Predictor = học dynamics trong latent space
Planner = sinh trajectory để lái
```

---

## 11. So Với NN-JEPA Hiện Tại

NN-JEPA hiện tại gần với phần nào?

Câu trả lời:

```text
NN-JEPA hiện tại gần với V-JEPA AC world-model hơn.
```

Sơ đồ NN-JEPA:

```text
images
    |
    v
V-JEPA encoder / feature cache
    |
    v
latent tokens
    |
    +-----------+
    |           |
    v           v
state         action
    |           |
    +-----concat/condition-----+
                               |
                               v
official-lite predictor / simple predictor
                               |
                               v
predicted future latent
```

Trong khi Drive-JEPA downstream là:

```text
images
    |
    v
V-JEPA encoder
    |
    v
latent / feature tokens
    |
    +----------------------------+
    |                            |
    v                            v
simple transformer decoder   proposal-centric planner
    |                            |
    v                            v
1 trajectory                 many trajectories + scorer
```

Nói ngắn:

```text
NN-JEPA hiện tại học latent future.
Drive-JEPA downstream học trajectory directly.
```

---

## 12. Câu Trả Lời Ngắn Gọn Nhất

Nếu bạn muốn nhớ thật ngắn:

```text
V-JEPA trong Drive-JEPA dùng để học encoder biểu diễn video lái xe.

Predictor của V-JEPA:
  dùng trong giai đoạn pretraining để dự đoán latent target/future.

Planner của Drive-JEPA:
  là module riêng, dùng feature từ encoder để sinh trajectory hoặc proposals.
```

Hoặc gói gọn hơn nữa:

```text
V-JEPA predictor học latent.
Drive-JEPA planner học lái.
```

---

## 13. Sơ Đồ Cuối Cùng Để Nhìn Một Phát

```text
                [A] V-JEPA PRETRAINING

video clip
   |
   +---------------------------+
   |                           |
   v                           v
context clip               target clip
   |                           |
   v                           v
encoder E_theta           EMA encoder E_theta_bar
   |                           |
   v                           v
context latent            target latent
   |                           ^
   v                           |
predictor P_phi ---------------+
   |
   v
predicted target latent


           [B] DRIVE-JEPA PERCEPTION-FREE PLANNER

2 front images --> pretrained V-JEPA encoder --> feature tokens
                                             \
ego status ----------------------------------> concat
                                                |
                                                v
                                      transformer decoder
                                                |
                                                v
                                          1 trajectory


         [C] DRIVE-JEPA PERCEPTION-BASED PROPOSAL PLANNER

2 front images --> V-JEPA image backbone --> image features
ego status --------------------------------> ego feature
learnable proposal embeddings -------------> initial proposal queries
                                             |
                                             v
                                   iterative proposal refinement
                                             |
                                             v
                                     32 trajectory proposals
                                             |
                                             v
                                           scorer
                                             |
                                             v
                                  momentum-aware selection
                                             |
                                             v
                                       final trajectory
```

---

## 14. Kết Luận

Drive-JEPA không phải:

```text
"lấy predictor V-JEPA rồi predictor đó trực tiếp lái xe"
```

Mà là:

```text
1. Dùng V-JEPA để học representation video
2. Lấy encoder đã học xong
3. Gắn planner riêng lên trên encoder đó
4. Planner này mới là phần sinh trajectory
```

Nếu cần, bước tiếp theo tôi có thể làm thêm một file thứ hai:

```text
so_do_so_sanh_drive_jepa_vs_nn_jepa.md
```

để đặt hai pipeline cạnh nhau, rất dễ thấy vì sao Drive-JEPA khác latent world-model hiện tại của bạn.

