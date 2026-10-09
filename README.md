# Lab Day 23 — Sensor Fusion: theo dõi xe bằng LiDAR + camera trên Waymo

> **Bài cá nhân** · **2 giờ trên lớp** + chuẩn bị ở nhà (CP0) ·
> **Deadline:** 23:59 ngày học lab, giờ Việt Nam (UTC+7), trừ khi key coach thông báo khác ·
> **Nộp:** repo `K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion` + link và commit hash trên LMS ([SUBMISSION.md](SUBMISSION.md))

## Tài liệu trong repo

| File | Đọc khi nào |
|---|---|
| [README.md](README.md) | Tổng quan, mục tiêu, chuẩn bị, cách bắt đầu (file này) |
| [CHECKPOINTS.md](CHECKPOINTS.md) | Trong giờ lab — việc cần làm và cách tự kiểm tra từng checkpoint |
| [SUBMISSION.md](SUBMISSION.md) | Đặt tên repo, file phải nộp, deadline, cách nộp và tự kiểm tra |
| [RUBRIC.md](RUBRIC.md) | Tiêu chí và điểm, bằng chứng cần có, điều kiện mất điểm, bonus |
| [RULES.md](RULES.md) | Quy định: làm cá nhân, dùng AI, sao chép, nộp muộn, bảo mật dữ liệu |
| [docs/HUONG_DAN_KY_THUAT.md](docs/HUONG_DAN_KY_THUAT.md) | Tra cứu: pipeline, thứ tự predict/update, API, schema metrics, xử lý lỗi cài đặt |
| [data/README.md](data/README.md) | Lấy dữ liệu Waymo và weights |
| [NOTICE.md](NOTICE.md) | Giấy phép và ghi nhận mã nguồn bên thứ ba |

---

## 1. Tổng quan

**Bài toán.** Xe tự hành cần biết các xe xung quanh đang ở đâu và đi về hướng nào,
liên tục qua thời gian. Một detector chỉ cho kết quả **từng frame**, có lúc bỏ sót,
có lúc báo nhầm. Bộ **tracker** nối các detection thành **track** có danh tính,
ước lượng cả vị trí lẫn vận tốc, và giảm nhiễu bằng cách kết hợp nhiều cảm biến.

**Hôm nay bạn xây gì.** Một tracker đa cảm biến chạy trên dữ liệu Waymo thật:

```text
LiDAR point cloud ─► BEV ─► FPN-ResNet ─► hộp 3D ─┐              (Part A–D: có sẵn)
                                                  ▼
      mỗi frame:  EKF predict ─► gán + update LiDAR ─► gán + update camera ─► quản lý track
                  (Part E)        (Part F)               (Part F + G)           (Part H)
                                                  ▼
                      metrics.json + grade_run.log: RMSE, ghost, miss  (Part I: chạy Waymo)
```

- Detector LiDAR (BEV + FPN-ResNet, weights có sẵn) đã viết xong — bạn **đọc hiểu** Part A–D.
- Bạn **viết** Part E–H: bộ lọc Kalman mở rộng (EKF), gán đo bằng Mahalanobis,
  mô hình đo camera, vòng đời track.
- Cuối buổi, bạn chạy cả pipeline trên một segment Waymo ở hai chế độ — **chỉ LiDAR**
  và **LiDAR + camera** — rồi giải thích sự khác biệt bằng số liệu.

**Thiết kế fusion: track-then-fuse.** Chỉ có **một** tracker. Mỗi frame, EKF predict
một lần, update bằng LiDAR, rồi update thêm bằng camera nếu xe nằm trong tầm nhìn
camera. Camera chỉ tinh chỉnh trạng thái; việc tạo, xác nhận và xoá track chỉ dựa vào LiDAR.

**Giới hạn cần biết.** Camera trong lab **không** chạy detector ảnh: platform lấy tâm
hộp 2D ground-truth của camera FRONT, thêm nhiễu theo `--seed`, rồi dùng làm đo cho
EKF. Kết quả fused vì thế không chứng minh chất lượng một camera detector.

## 2. Mục tiêu học tập và cách đo

Sau buổi lab, bạn **làm được** những việc dưới đây. Cột cuối là bằng chứng dùng để đo
mức đạt; điểm chi tiết ở [RUBRIC.md](RUBRIC.md).

| # | Mục tiêu (bạn có thể…) | Đo bằng | Mức đạt |
|---|---|---|---|
| 1 | Cài **EKF 6D** `(px, py, pz, vx, vy, vz)` vận tốc không đổi: `F`, `Q`, predict, update | Test Part E (`test_kalman.py` + test chấm) | Pass 100% test Part E |
| 2 | Gán đo vào track bằng **Mahalanobis + cổng χ²** và gán greedy | Test Part F | Pass 100% test Part F |
| 3 | Viết **mô hình đo camera**: kiểm tra FOV, chiếu pinhole `h(x)`, từ chối điểm không hợp lệ | Test Part G | Pass 100% test Part G |
| 4 | Quản lý **vòng đời track**: khởi tạo, cộng/trừ score, xác nhận, xoá — chỉ theo LiDAR | Test Part H + vấn đáp | Pass 100% test Part H; giải thích được vì sao camera không đổi score |
| 5 | Chạy tracker trên Waymo và đạt chất lượng tracking chuẩn | `metrics.json` | RMSE LiDAR và fused ≤ 0.45 m; `precision_track` ≥ 0.75; `coverage` ≥ 0.70 |
| 6 | Chứng minh camera **không làm tracking xấu đi** | `metrics.json` | `rmse_fused − rmse_lidar` ≤ 0.05 m |
| 7 | Phân biệt **detection** và **tracking**; giải thích **track-then-fuse** trên log | Câu 3, 5 trong `student/SUBMISSION.md` | Trả lời đúng, dẫn tới log hoặc code |
| 8 | Đọc RMSE **cùng** số cặp ghép, ghost, miss — không kết luận từ một con số | Phần tóm tắt kết quả trong `student/SUBMISSION.md` | Giải thích khác biệt hai mode bằng ít nhất RMSE + matches + ghost/miss, số khớp `metrics.json` |

`precision_track = matches / (matches + ghost_track_frames)`, `coverage = matches / det_tp`
— xem [RUBRIC.md](RUBRIC.md) mục 1.2.

## 3. Bạn làm gì

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

## 4. Chuẩn bị trước buổi học (CP0 — làm ở nhà)

| Việc | Ghi chú |
|---|---|
| Tải 1 segment Waymo `.tfrecord` + weights `fpn_resnet_18_epoch_300.pth` | Từ [Google Drive Day23-Fusion](https://drive.google.com/drive/folders/1XTXA60c9gV6rkK6fWohMyHw8qpsfYBlE?usp=sharing); segment mặc định ghi trong [data/README.md](data/README.md) |
| Python 3.12 (conda, uv hoặc pip) hoặc Docker | Mục 5, bước 2 |
| Đọc §2 của [docs/HUONG_DAN_KY_THUAT.md](docs/HUONG_DAN_KY_THUAT.md) | Sơ đồ pipeline và thứ tự predict → AssocL → AssocC |
| Đọc lướt Part A–D trong `student/workspace/` | Biết detector trả gì cho tracker |

Không tải dữ liệu qua mạng lớp trong giờ lab.

## 5. Bắt đầu

Mọi lệnh chạy từ **gốc repo**.

**Bước 1 — Fork và clone.** Bấm **Fork**, ở ô *Repository name* đặt tên
`K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion` (ví dụ
`K4-L2L3-DAY23-NguyenVanA-2A20260000-SensorFusion`; quy tắc ở [SUBMISSION.md](SUBMISSION.md) mục 2), rồi:

```bash
git clone https://github.com/<tai-khoan>/K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion.git
cd K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion
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

**Bước 3 — Biến môi trường và cấu hình.** Lab **không cần API key**. Các biến môi
trường được liệt kê, có giải thích, trong [`.env.example`](.env.example):

```bash
cp .env.example .env
set -a; source .env; set +a      # chạy lại mỗi khi mở terminal mới
cp student/config/paths.example.yaml student/config/paths.yaml
```

Đặt dữ liệu vào `data/Waymo/` và weights vào `data/weights/`, hoặc sửa đường dẫn
trong `student/config/paths.yaml`. `.env` và `paths.yaml` không được commit.

**Bước 4 — Kiểm tra:**

```bash
pytest student/tests -q
fusion-run-lab --help
```

Kết quả đúng khi chưa làm bài: không có dòng `failed`; các test E–H báo `xfailed`
vì hàm còn `NotImplementedError`. Khi bạn implement xong một Part, test của Part đó
phải chuyển sang `passed`.

## 6. Lịch 2 giờ

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

## 7. Cấu trúc repo

```text
.
├── README.md, CHECKPOINTS.md, SUBMISSION.md, RUBRIC.md, RULES.md, NOTICE.md
├── .env.example                 # biến môi trường (không có key)
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
    ├── bonus/                   # bằng chứng bonus (không bắt buộc)
    └── SUBMISSION.md            # báo cáo nộp bài
```

## 8. Tài liệu tham khảo

- Waymo Open Dataset — Perception v1: <https://waymo.com/open/>
- Thrun, Burgard, Fox — *Probabilistic Robotics*, chương 3 (Kalman, EKF)
- Bar-Shalom, Li, Kirubarajan — *Estimation with Applications to Tracking and Navigation* (gating, association)
- Mã nguồn, starter course và giấy phép: [NOTICE.md](NOTICE.md), [§10 Acknowledgement](#10-acknowledgement)

## 9. Khi gặp khó

- Kẹt cùng một lỗi quá **10 phút** thì hỏi lab coach.
- Lỗi cài đặt, `fusion_lab` không import được, Docker: §3 của [hướng dẫn kỹ thuật](docs/HUONG_DAN_KY_THUAT.md).
- Test fail mà không rõ vì sao: chạy riêng một test với `-x -vv`, đọc assertion, đối chiếu gợi ý `# vi: TODO`.
- Chưa có dữ liệu Waymo đúng giờ: làm CP1–CP4 bằng `pytest` trước (không cần dữ liệu), chạy CP5 khi có dữ liệu.

## 10. Acknowledgement

Lab và pipeline detector LiDAR dựa trên các dự án mã nguồn mở. Chi tiết giấy phép và phần nào được vendored trong repo: [NOTICE.md](NOTICE.md).

| Dự án | Vai trò trong lab |
|---|---|
| [Simple Waymo Open Dataset Reader](https://github.com/gdlg/simple-waymo-open-dataset-reader) | Đọc segment Waymo (`.tfrecord`) nhẹ, không TensorFlow/Bazel — vendored trong `platform/third_party/waymo_reader/` |
| [SFA3D](https://github.com/maudzung/SFA3D) — *Super Fast and Accurate 3D Object Detection based on 3D LiDAR Point Clouds* | FPN-ResNet + BEV detector (weights tải riêng) — vendored trong `platform/third_party/objdet_models/resnet/` |
| [Complex-YOLOv4-PyTorch](https://github.com/maudzung/Complex-YOLOv4-Pytorch) — *Real-time 3D Object Detection on Point Clouds* | Detector BEV thay thế trong starter Udacity; **lab này không vendored** — chỉ dùng nhánh SFA3D/FPN-ResNet |
| [SDCND: Sensor Fusion and Tracking](https://github.com/udacity/nd013-c2-fusion-starter) | Starter Udacity Self-Driving Car Engineer Nanodegree (Course 2) — cùng hướng bài (detection + EKF + fusion); **không** phân phối lại mã starter trong repo VinUni |
