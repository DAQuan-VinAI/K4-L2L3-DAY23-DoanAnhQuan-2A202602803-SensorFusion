# Hướng dẫn nộp bài

Nộp bài = **push lên fork của bạn** + **nộp link repo và commit hash trên LMS**.
Deadline theo thông báo trên LMS.

## 1. Fork đúng tên

Tên repo fork: `<HoVaTen>-<MSSV>-Track4-Day23`, viết liền không dấu, ví dụ
`NguyenVanA-20240123-Track4-Day23`. Fork để **Public** (hoặc Private và thêm
giảng viên làm collaborator nếu lớp yêu cầu). Cách tạo fork: [README.md](README.md) mục 4.

## 2. Những gì phải có trong repo

| File | Nội dung |
|---|---|
| `student/workspace/kalman.py` | Part E |
| `student/workspace/association.py` | Part F |
| `student/workspace/camera_fusion.py` | Part G |
| `student/workspace/track_management.py` | Part H |
| `student/artifacts/metrics.json`, `grade_run.log` | Kết quả lần chạy chấm điểm |
| `student/artifacts/metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`, `grade_run_fused.log` | File theo từng mode, sinh cùng lần chạy |
| `student/SUBMISSION.md` | Báo cáo, đã điền đủ, có khai báo AI |

**Không** được có: dữ liệu `.tfrecord`, weights `.pth`, `student/config/paths.yaml`,
file nén, file > 20 MB, API key.

## 3. Lần chạy chấm điểm

Trong `student/config/paths.yaml`: `frame_start: 0`, `frame_end: 198`, segment mặc định.
Từ gốc repo:

```bash
export DAY23_STUDENT_ROOT="$(pwd)/student"
pytest student/tests -q
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

- `pytest` không còn `failed` hay `xfailed`.
- Chạy lại **sau lần sửa code cuối cùng**, để artifacts khớp với code nộp.
- Không sửa tay bất kỳ file nào trong `student/artifacts/`.

## 4. Tự kiểm tra

```bash
python tools/check_submission.py
```

Sửa đến khi dòng cuối là `KẾT QUẢ: SẴN SÀNG NỘP`. Công cụ kiểm tra cấu trúc,
artifacts, `SUBMISSION.md`, file cấm và key bị lộ; nó **không** chấm điểm.

## 5. Push và nộp trên LMS

```bash
git add student/workspace student/artifacts student/SUBMISSION.md
git commit -m "CP6: report and final run"
git push origin main
git rev-parse HEAD
```

Trên LMS, nộp:

1. Link fork, ví dụ `https://github.com/<tai-khoan>/NguyenVanA-20240123-Track4-Day23`.
2. Commit hash từ `git rev-parse HEAD` (40 ký tự).

Kiểm tra lại: mở link fork trên trình duyệt, đúng commit hash, thấy `metrics.json`
và `SUBMISSION.md` đã điền.

## 6. Sau khi nộp

- Không `push --force` hay sửa lịch sử commit ([RULES.md](RULES.md) mục 5).
- Giảng viên có thể gọi vấn đáp 5 phút ([RUBRIC.md](RUBRIC.md) mục 4).
