# Hướng dẫn nộp bài

> **Bài cá nhân.** Mỗi học viên tự nộp repo của mình.
> File này là **hướng dẫn nộp**. Báo cáo bạn phải điền nằm ở [`student/SUBMISSION.md`](student/SUBMISSION.md).

## 1. Tóm tắt

| Mục | Quy định |
|---|---|
| Tên repo nộp | `K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion` |
| Nơi nộp | Repo GitHub của bạn **+** link repo và commit hash trên **LMS** |
| Deadline | **23:59 ngày học lab, giờ Việt Nam (UTC+7)**, trừ khi key coach thông báo deadline khác trong vòng 48 giờ sau buổi lab |
| Nộp muộn | Trừ điểm theo [RULES.md](RULES.md) mục 4 |
| Tự kiểm tra | `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP` |

## 2. Đặt tên repo

```text
K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion
```

- Họ tên viết **không dấu, không khoảng trắng**, viết hoa chữ cái đầu mỗi từ.
- Các phần ngăn cách bằng dấu `-`; ngày học viết hai chữ số (`DAY23`).
- Ví dụ: `K4-L2L3-DAY23-NguyenVanA-2A20260000-SensorFusion`.

Sai tên repo bị trừ điểm theo [RUBRIC.md](RUBRIC.md) mục 3.

Tạo repo bằng **Fork** và điền tên trên ngay ở trang tạo fork (đã fork rồi thì đổi tên
trong *Settings → Repository name*). Repo fork để Public. Chi tiết: [README.md](README.md) mục 5.

## 3. Cấu trúc repo và các file phải nộp

```text
K4-L2L3-DAY23-<HoVaTen>-<MSSV>-SensorFusion/
└── student/
    ├── workspace/
    │   ├── kalman.py               # Part E — bắt buộc
    │   ├── association.py          # Part F — bắt buộc
    │   ├── camera_fusion.py        # Part G — bắt buộc
    │   └── track_management.py     # Part H — bắt buộc
    ├── artifacts/
    │   ├── metrics.json            # bắt buộc — lần chạy chấm điểm
    │   ├── grade_run.log           # bắt buộc
    │   ├── metrics_lidar.json      # bắt buộc — sinh cùng lần chạy
    │   ├── metrics_fused.json
    │   ├── grade_run_lidar.log
    │   └── grade_run_fused.log
    ├── bonus/                      # không bắt buộc — bằng chứng bonus (RUBRIC mục 2)
    └── SUBMISSION.md               # bắt buộc — báo cáo + khai báo AI
```

Giữ nguyên các file còn lại của đề. **Không** đổi tên hay di chuyển các file trên:
grader tìm đúng đường dẫn này.

**Không được có trong repo:** dữ liệu Waymo `.tfrecord`, weights `.pth`,
`student/config/paths.yaml`, `.env`, file nén, file > 20 MB, API key/token.

## 4. Lần chạy chấm điểm

Trong `student/config/paths.yaml`: `frame_start: 0`, `frame_end: 198`, segment mặc định.
Từ gốc repo:

```bash
pytest student/tests -q
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

- `pytest` không còn `failed` hay `xfailed`.
- Chạy lại **sau lần sửa code cuối cùng**, để artifacts khớp với code nộp.
- Không sửa tay bất kỳ file nào trong `student/artifacts/`.

## 5. Kiểm tra trước khi nộp

```bash
python tools/check_submission.py
```

Sửa đến khi dòng cuối là `KẾT QUẢ: SẴN SÀNG NỘP`. Công cụ kiểm tra:

- Part E–H không còn hàm `NotImplementedError("TODO ...")`.
- Đủ 6 file artifacts đã commit; `metrics.json` khớp `grade_run.log`, chạy `compare`, có cả hai mode, `seed = 0`.
- `student/SUBMISSION.md` đã điền họ tên, MSSV, link repo, khai báo AI.
- Không có file cấm, file > 20 MB, key bị lộ; mọi thay đổi trong `student/` đã commit.

Công cụ **không** chấm điểm. Tự rà thêm checklist cuối `student/SUBMISSION.md`.

## 6. Push và nộp trên LMS

```bash
git add student/workspace student/artifacts student/SUBMISSION.md
git commit -m "CP6: report and final run"
git push origin main
git rev-parse HEAD
```

Trên LMS, nộp:

1. Link repo, ví dụ `https://github.com/<tai-khoan>/K4-L2L3-DAY23-NguyenVanA-2A20260000-SensorFusion`.
2. Commit hash từ `git rev-parse HEAD` (40 ký tự).

Mở link repo trên trình duyệt để kiểm tra: đúng commit hash, thấy `metrics.json`
và `student/SUBMISSION.md` đã điền.

## 7. Sau khi nộp

- Bài được chấm ở commit cuối cùng trước deadline ([RULES.md](RULES.md) mục 5).
- Không `push --force` hay sửa lịch sử commit.
- Giảng viên có thể gọi vấn đáp 5 phút ([RUBRIC.md](RUBRIC.md) mục 4).
