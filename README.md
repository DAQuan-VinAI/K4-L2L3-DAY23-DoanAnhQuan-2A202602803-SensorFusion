# Lab Day 23 — Sensor Fusion: theo dõi xe bằng LiDAR + camera trên Waymo

Bạn hoàn thiện một tracker đa cảm biến chạy trên dữ liệu Waymo thật. Detector LiDAR
(BEV + FPN-ResNet) đã có sẵn; bạn viết **bộ lọc Kalman mở rộng (EKF)**, **gán đo
lường bằng Mahalanobis**, **mô hình đo camera** và **vòng đời track**, rồi so sánh
tracking chỉ dùng LiDAR với tracking có thêm camera.

> **Thời lượng:** 2 giờ trên lớp + phần chuẩn bị ở nhà (CP0). **Làm cá nhân.**
> **Deadline:** theo thông báo trên LMS.

## Tài liệu trong repo

| File | Đọc khi nào |
|---|---|
| [README.md](README.md) | Tổng quan, chuẩn bị, cách bắt đầu (file này) |
| [CHECKPOINTS.md](CHECKPOINTS.md) | Trong giờ lab — việc cần làm và cách tự kiểm tra từng checkpoint |
| [docs/HUONG_DAN_KY_THUAT.md](docs/HUONG_DAN_KY_THUAT.md) | Tra cứu: pipeline, thứ tự predict/update, API, schema metrics, xử lý lỗi cài đặt |
| [RUBRIC.md](RUBRIC.md) | Cách tính điểm |
| [RULES.md](RULES.md) | Quy định: làm cá nhân, dùng AI, nộp muộn, dữ liệu Waymo |
| [NOP_BAI.md](NOP_BAI.md) | Cách nộp bài: fork + LMS |
| [data/README.md](data/README.md) | Lấy dữ liệu Waymo và weights |

---

## 1. Mục tiêu học tập

Sau buổi lab, bạn có thể:

1. Phân biệt **detection** (đo độc lập từng frame) và **tracking** (giữ danh tính qua thời gian).
2. Viết **EKF 6D** `(px, py, pz, vx, vy, vz)` với mô hình vận tốc không đổi: `F`, `Q`, predict, update.
3. Gán đo vào track bằng **khoảng cách Mahalanobis + cổng χ²**, rồi gán greedy.
4. Viết **mô hình đo camera**: kiểm tra FOV, chiếu pinhole `h(x)`, xử lý điểm không hợp lệ.
5. Quản lý **vòng đời track**: khởi tạo, cộng/trừ score, xác nhận, xoá.
6. Giải thích **track-then-fuse**: một tracker, predict một lần mỗi frame, update LiDAR rồi update camera.
7. Đọc RMSE cùng số cặp ghép, ghost và miss — không kết luận chỉ từ một con số.

## 2. Bạn làm gì

| Part | File trong `student/workspace/` | Việc | Bạn viết? |
|---|---|---|---|
| A | `lidar_viz.py` | Range image | Không — đọc hiểu |
| B | `bev_mapping.py` | Point cloud → BEV | Không — đọc hiểu |
| C | `detection_pipeline.py` | FPN-ResNet → hộp 3D | Không — đọc hiểu |
| D | `detection_metrics.py` | IoU, precision/recall | Không — đọc hiểu |
| **E** | `kalman.py` | EKF predict/update | **Có** |
| **F** | `association.py` | Mahalanobis, gating, gán LiDAR/camera | **Có** |
| **G** | `camera_fusion.py` | FOV, `h(x)`, nhiễu đo camera | **Có** |
| **H** | `track_management.py` | Khởi tạo, score, xoá track | **Có** |
| I | `fusion-run-lab` (platform) | Chạy cả pipeline trên Waymo | Không — chỉ chạy |

Mỗi hàm cần viết có `raise NotImplementedError("TODO: ...")` và gợi ý `# vi: TODO Part …`
ngay trong file. **Giữ nguyên tên và chữ ký hàm**: khi chấm, giảng viên chạy bộ test
gốc trên `workspace/` của bạn.

Camera trong lab **không** chạy detector ảnh: platform lấy tâm hộp 2D ground-truth
của camera FRONT, thêm nhiễu theo `--seed`, rồi dùng làm đo cho EKF. Kết quả fused
vì thế không chứng minh chất lượng một camera detector.

## 3. Chuẩn bị trước buổi học (CP0 — làm ở nhà)

| Việc | Ghi chú |
|---|---|
| Đăng ký Waymo Open Dataset, chấp nhận điều khoản | Bắt buộc trước khi nhận dữ liệu — xem [data/README.md](data/README.md) |
| Tải 1 segment Waymo `.tfrecord` + weights `fpn_resnet_18_epoch_300.pth` | Segment mặc định ghi trong [data/README.md](data/README.md) |
| Python 3.12 (conda, uv hoặc pip) hoặc Docker | Mục 4, bước 2 |
| Đọc §2 của [docs/HUONG_DAN_KY_THUAT.md](docs/HUONG_DAN_KY_THUAT.md) | Sơ đồ pipeline và thứ tự predict → AssocL → AssocC |
| Đọc lướt Part A–D trong `student/workspace/` | Biết detector trả gì cho tracker |

Không tải dữ liệu qua mạng lớp trong giờ lab.

## 4. Bắt đầu

Mọi lệnh chạy từ **gốc repo**.

**Bước 1 — Fork và clone.** Fork repo này về tài khoản GitHub của bạn, đặt tên
`<HoVaTen>-<MSSV>-Track4-Day23` (ví dụ `NguyenVanA-20240123-Track4-Day23`), rồi:

```bash
git clone https://github.com/<tai-khoan>/<HoVaTen>-<MSSV>-Track4-Day23.git
cd <HoVaTen>-<MSSV>-Track4-Day23
git remote add upstream https://github.com/VinUni-AI20k/K4-Track4-Day23-Sensor-Fusion-Student.git
```

Khi giảng viên báo có cập nhật đề: `git pull upstream main`.

**Bước 2 — Cài môi trường** (chọn một cách):

```bash
conda env create -f environment.yml
conda activate day23_sensor_fusion
```

```bash
uv venv -p 3.12 && source .venv/bin/activate
uv pip install -e platform/third_party/waymo_reader -e platform -e student pytest
```

Trên macOS, nếu repo nằm trong `~/Documents` hoặc `~/Desktop` mà vẫn báo
`No module named 'fusion_lab'`, đặt venv ở ngoài các thư mục đó — xem §3 của
[hướng dẫn kỹ thuật](docs/HUONG_DAN_KY_THUAT.md). Docker: cũng ở §3.

**Bước 3 — Biến môi trường và cấu hình:**

```bash
export DAY23_STUDENT_ROOT="$(pwd)/student"
export FUSION_LAB_PLATFORM="$(pwd)/platform"
cp student/config/paths.example.yaml student/config/paths.yaml
```

Đặt dữ liệu vào `data/Waymo/` và weights vào `data/weights/`, hoặc sửa đường dẫn
trong `student/config/paths.yaml`. File này không được commit.

**Bước 4 — Kiểm tra:**

```bash
pytest student/tests -q
fusion-run-lab --help
```

Kết quả đúng khi chưa làm bài: không có dòng `failed`; các test E–H báo `xfailed`
vì hàm còn `NotImplementedError`. Khi bạn implement xong một Part, test của Part đó
phải chuyển sang `passed`.

## 5. Lịch 2 giờ

| Thời gian | Checkpoint | Nội dung | Sản phẩm |
|---|---|---|---|
| Ở nhà | CP0 | Fork, cài đặt, dữ liệu, weights, đọc Part A–D | `pytest` chạy, không có `failed` |
| 0:00 – 0:25 | CP1 | Part E — EKF | `test_kalman.py` pass |
| 0:25 – 0:50 | CP2 | Part G — mô hình đo camera | Test camera pass |
| 0:50 – 1:15 | CP3 | Part F — association | Test association pass |
| 1:15 – 1:35 | CP4 | Part H — vòng đời track | Toàn bộ `student/tests` pass |
| 1:35 – 1:50 | CP5 | Chạy Waymo `--fusion compare` | `metrics.json`, `grade_run.log` |
| 1:50 – 2:00 | CP6 | Viết `SUBMISSION.md`, tự kiểm tra, push | `check_submission.py` báo sẵn sàng |

Chi tiết từng checkpoint: [CHECKPOINTS.md](CHECKPOINTS.md). Commit sau mỗi
checkpoint với message `CPx: <việc vừa làm>`.

## 6. Cấu trúc repo

```text
.
├── README.md, CHECKPOINTS.md, RUBRIC.md, RULES.md, NOP_BAI.md
├── docs/HUONG_DAN_KY_THUAT.md   # pipeline, API, metrics, xử lý sự cố
├── data/README.md               # hướng dẫn dữ liệu (dữ liệu thật không commit)
├── environment.yml, docker/     # môi trường
├── platform/                    # runtime fusion_lab + fusion-run-lab (không sửa)
├── tools/check_submission.py    # tự kiểm tra trước khi nộp
└── student/
    ├── workspace/               # Part A–D đọc hiểu, Part E–H bạn viết
    ├── tests/                   # bộ tự kiểm tra (gồm test E–H dùng khi chấm)
    ├── config/                  # paths.example.yaml → paths.yaml (không commit)
    ├── artifacts/               # metrics.json, grade_run.log (phải commit)
    └── SUBMISSION.md            # báo cáo nộp bài
```

## 7. Tài liệu tham khảo

- Waymo Open Dataset — Perception v1: <https://waymo.com/open/>
- SFA3D (detector BEV + FPN-ResNet): <https://github.com/maudzung/SFA3D>
- Thrun, Burgard, Fox — *Probabilistic Robotics*, chương 3 (Kalman, EKF)
- Bar-Shalom, Li, Kirubarajan — *Estimation with Applications to Tracking and Navigation* (gating, association)
- Nguồn và giấy phép thành phần bên thứ ba: [NOTICE.md](NOTICE.md)

## 8. Khi gặp khó

- Kẹt cùng một lỗi quá **10 phút** thì hỏi lab coach.
- Lỗi cài đặt, `fusion_lab` không import được, Docker: §3 của [hướng dẫn kỹ thuật](docs/HUONG_DAN_KY_THUAT.md).
- Test fail mà không rõ vì sao: chạy riêng một test với `-x -vv`, đọc assertion, đối chiếu gợi ý `# vi: TODO`.
- Chưa có dữ liệu Waymo đúng giờ: làm CP1–CP4 bằng `pytest` trước (không cần dữ liệu), chạy CP5 khi có dữ liệu.
