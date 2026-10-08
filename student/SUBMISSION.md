# Nộp bài — Day 23 Sensor Fusion Lab

## Thông tin học viên

- Họ tên:
- Email:
- Ngày nộp:

## Tóm tắt kết quả

- Detection precision / recall (baseline Part A–D có sẵn, từ `artifacts/metrics.json`):
- Tracking lidar-only vs fused (bắt buộc — RMSE hoặc mô tả log):
- Fusion mode đã chạy (`--fusion compare` khuyến nghị):

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
2. Vì sao cần gating Mahalanobis trước khi gán?
3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

## AI / coding assistant (nếu có)

- Phần nào dùng assistant:
- Cách bạn đã kiểm tra lại (pytest / run_lab):

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement (TODO `# vi:`)
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] `artifacts/grade_run.log`
- [ ] `artifacts/metrics.json`
- [ ] File SUBMISSION.md này
- [ ] Không nộp Waymo tfrecord / weights
