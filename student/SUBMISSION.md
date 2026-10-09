# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Doan Anh Quan
- MSSV: 2A202602803
- Email: doananhquan2607@gmail.com
- Link repo (fork): https://github.com/DAQuan-VinAI/K4-L3-Day23-DoanAnhQuan-2A202602803
- Commit hash nộp (`git rev-parse HEAD`): hash của commit CP6 (chứa file này) nộp trên LMS; code và artifacts chấm điểm nằm ở commit CP5 `f6c622f`

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3436, 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668, 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Hai mode có cùng `matches` (502), `ghost_track_frames` (0), `missed_gt_frames` (239) và `mean_confirmed_tracks`; trong `grade_run.log` các cột `confirmed`, `matches`, `ghosts`, `misses` trùng nhau ở cả 199 frame. Vì vậy hai RMSE được tính trên cùng một tập cặp track–GT và so sánh trực tiếp được. Khác biệt duy nhất là `sum_sq_err`: 11.34 (lidar) so với 9.27 (fused), tức RMSE giảm 0.0145 m (khoảng 10%) khi thêm update camera. Mức giảm này không đều theo frame: trong 195 frame có cặp ghép, fused có sai số nhỏ hơn ở 137 frame và lớn hơn ở 58 frame, nên camera cải thiện trung bình chứ không phải mọi lúc. Số ghost bằng 0 khớp với precision detection cao (0.97) và việc phải đủ hit lidar mới confirmed. 239 lượt miss không do camera gây ra hay sửa được: camera không tạo track, nên miss đến từ recall của detector lidar (0.70, `fn` = 222) và từ các frame đầu khi track chưa confirmed (frame 0–3 chưa có track confirmed nào).

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   Lidar cho `z` 3×1 là vị trí `[x, y, z]` tính bằng mét trong hệ cảm biến. Hàm đo
   `h(x) = R p + t` tuyến tính, nên `H` là ma trận 3×6 cố định (khối xoay và ba cột
   vận tốc bằng 0), `R = diag(0.1²)` m². Camera cho `z` 2×1 là pixel `[u, v]`. Hàm đo
   là phép chiếu pinhole `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s`, phi tuyến vì
   chia cho độ sâu, nên `H` 2×6 là Jacobian phải tính lại tại từng state, và
   `R = diag(5²)` pixel². Camera không quan sát được độ sâu: một phép đo chỉ ràng buộc
   hướng nhìn tới vật, còn khoảng cách vẫn dựa vào lidar. Phép chiếu cũng chỉ xác định
   khi tọa độ hữu hạn và `x_s > 1e-6`, nên phải kiểm tra FOV trước khi chiếu.

2. Vì sao cần gating Mahalanobis trước khi gán?

   Khoảng cách `d² = γᵀ S⁻¹ γ` chuẩn hóa innovation theo độ bất định `S = H P Hᵀ + R`,
   nên so sánh được giữa track chắc chắn và track còn mơ hồ, giữa mét của lidar và
   pixel của camera. Dưới giả thiết Gauss, `d²` tuân theo phân phối χ² với số bậc tự do
   bằng số chiều đo, nên ngưỡng `chi2.ppf(0.995, dim_meas)` giữ lại 99.5% phép đo đúng
   và loại phần còn lại. Nếu không gate, thuật toán greedy vẫn sẽ ghép một track với
   phép đo gần nhất dù nó ở rất xa: track bị kéo sang vật khác, phép đo đó không còn
   để sinh track mới, và track lẽ ra bị miss lại được cộng score. Gate cũng làm ma
   trận chi phí thưa đi, các cặp bị loại mang giá trị `inf`.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Track-then-fuse. Chỉ có một danh sách track, mỗi frame predict một lần, rồi update
   lần lượt theo từng cảm biến: lidar trước (kèm quản lý vòng đời), camera sau. Hai
   loại đo không bị gộp thành một phép đo chung trước khi vào tracker. Dấu vết trên
   log: `grade_run_lidar.log` và `grade_run_fused.log` có `confirmed`, `matches`,
   `ghosts`, `misses` giống nhau ở cả 199 frame, chỉ `sum_sq_err` khác (ví dụ frame
   100: 0.0601 so với 0.0488, cùng 3 track confirmed và 3 cặp ghép). Nghĩa là camera
   chỉ chỉnh state của những track mà lidar đã dựng, không thay đổi tập track. Nếu là
   fuse-then-track thì detection đầu vào đã khác, và số track/số cặp ghép giữa hai
   mode sẽ khác theo.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Innovation camera `γ = z − h(x)` không còn dao động quanh 0 mà có độ lệch hệ thống:
   cùng dấu, cùng hướng qua nhiều frame và nhiều track. Lệch góc xoay cho độ lệch pixel
   gần như không đổi theo khoảng cách; lệch tịnh tiến cho độ lệch lớn ở vật gần và
   nhỏ ở vật xa. Hệ quả là `d²` của camera lớn bất thường so với kỳ vọng χ² hai bậc tự
   do, nhiều cặp bị gate loại nên phép đo camera không ghép được. Các cặp vẫn lọt qua
   gate thì kéo state lệch ngang, làm innovation lidar ở frame sau lệch theo hướng
   ngược lại. Trên metrics, RMSE fused sẽ tăng lên so với lidar-only thay vì giảm như
   ở lần chạy này.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng thì không có `meas.sensor` nào để suy ra đây là lượt lidar hay
   camera, trong khi `manage_tracks` xử lý hai lượt khác nhau. Một frame lidar không
   có detection vẫn là thông tin: mọi track trong FOV đều bị miss, phải trừ score và
   xét xóa. Nếu bỏ qua bước quản lý ở frame rỗng, track của vật đã biến mất sẽ không
   bao giờ bị trừ điểm. Lidar được giao quyền quyết định tồn tại vì nó đo vị trí 3D
   đầy đủ: đủ để khởi tạo state, và việc không thấy vật trong FOV là bằng chứng đáng
   tin. Camera chỉ có FOV phía trước và không có độ sâu, nên không khởi tạo được vị
   trí 3D từ một phép đo; track nằm ngoài FOV camera cũng không thể coi là miss. Nếu
   cả hai cảm biến cùng cộng/trừ score thì mỗi frame track bị tính hai lần, và các
   ngưỡng theo `window` mất ý nghĩa.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Track mới có score `1/window = 1/6`, trạng thái `initialized`. Mỗi hit lidar cộng
   `1/6` (tối đa 1) và chuyển track chưa confirmed sang `tentative`; mỗi miss lidar
   trong FOV trừ `1/6`. Track thành `confirmed` khi `score > confirmed_threshold = 0.8`,
   tức cần score `5/6`: khởi tạo cộng bốn hit liên tiếp. Điều này khớp log: track sinh
   ở frame 0 và confirmed đầu tiên xuất hiện ở frame 4. Track đã confirmed không bị hạ
   trạng thái khi miss, chỉ bị trừ score. Xóa track khi một trong các điều kiện sau
   đúng: `P[0,0]` hoặc `P[1,1]` lớn hơn `max_P = 9` m² bất kể score; track confirmed
   có `score < delete_threshold = 0.6` (từ score 1 là sau ba miss liên tiếp: 5/6, 4/6,
   rồi 3/6); track chưa confirmed có `score <= 0`. Lượt camera không xét điều kiện nào
   trong số này.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude)
- Dùng cho phần nào (hàm, câu hỏi, debug): Viết code toàn bộ các hàm Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); tải dữ liệu và chạy `fusion-run-lab`; soạn bản nháp báo cáo này, gồm phần tóm tắt kết quả và sáu câu giải thích.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` báo 128 passed, không còn `failed`/`xfailed`; chạy Waymo đủ frame 0–198 với `--fusion compare --seed 0`; số liệu trong báo cáo chép từ `metrics.json` của lần chạy đó; đối chiếu `grade_run.log` với `metrics.json` (398 record, mỗi `(mode, frame)` một record, hai đẳng thức `matches+ghosts==confirmed` và `matches+misses==valid_gt` đúng trên mọi record, tổng khớp metrics); đối chiếu công thức trong code với `docs/HUONG_DAN_KY_THUAT.md` mục 2. Sau khi cập nhật lại thư mục `data/` (đủ 4 segment và weights), chạy lại `pytest` (128 passed) và lần chạy chấm điểm với cùng segment, khoảng frame và seed: sáu file trong `student/artifacts/` trùng từng byte với bản đã commit ở CP5 (`git diff` rỗng), nên số liệu trong báo cáo không đổi.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
