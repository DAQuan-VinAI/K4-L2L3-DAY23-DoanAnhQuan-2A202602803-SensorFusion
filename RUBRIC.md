# Rubric chấm điểm

Bài lab làm **cá nhân**. Điểm của mỗi học viên:

- **Phần chính: 100 điểm** — A (15) + B (60) + C (25), mục 1.
- **Bonus: tối đa +5**, mục 2, cộng ngoài 100 điểm chính.
- **Trừ điểm và mất điểm:** mục 3. **Vấn đáp xác minh:** mục 4.

Part A–D là code có sẵn, **không** chấm phần sửa các file đó.

---

## 1. Phần chính (100 điểm)

| Hạng mục | Điểm | Chấm | Bằng chứng |
|---|---|---|---|
| A — Tích hợp Waymo | 15 | Tự động | `student/artifacts/metrics.json`, `grade_run.log` |
| B — Tracking + fusion (E–H) | 60 = 45 tự động + 15 thủ công | Tự động + giảng viên | `student/workspace/` E–H, artifacts |
| C — Báo cáo | 25 | Giảng viên | `student/SUBMISSION.md` |

### 1.1 A — Tích hợp Waymo (15 điểm, tự động)

Đạt đủ 15 điểm khi:

- `metrics.json` có `fusion_mode = "compare"`, có kết quả cả `tracking.lidar` và `tracking.fused`, có `seed` (số nguyên) và `segment`.
- `grade_run.log` có **đúng một** record cho mỗi `(mode, frame)` trong khoảng `frames` (tính cả hai đầu).
- Mọi tổng trong `metrics.json` khớp khi tính lại từ `grade_run.log`; mỗi record thoả `matches + ghosts == confirmed` và `matches + misses == valid_gt`.

Precision/recall của detection chỉ được **ghi nhận** (detector có sẵn), không tính điểm.

### 1.2 B — Tracking + fusion (60 điểm)

**Tự động (45 điểm)** gồm ba phần, tính từ log của bạn:

| Phần | Tối đa | Cách tính |
|---|---|---|
| Fused | 20 | phần nguyên của `20 × q_fused` |
| LiDAR | 15 | phần nguyên của `15 × q_lidar` |
| Fusion nhất quán | 10 | phần nguyên của `10 × q_fused` nếu `rmse_fused − rmse_lidar ≤ 0.05 m`, ngược lại 0 |

Hệ số chất lượng mỗi mode:

```text
q = hệ_số_RMSE × min(1, precision_track / 0.75) × min(1, coverage / 0.70)

precision_track = matches / (matches + ghost_track_frames)
coverage        = matches / det_tp
```

| RMSE (m) | Hệ số RMSE |
|---|---|
| ≤ 0.45 | 1 |
| > 0.45 và ≤ 0.75 | 0.5 |
| > 0.75 hoặc `null` | 0 |

- RMSE là sai số vị trí **3D** của các cặp confirmed track – xe thật được ghép một-một trong cổng XY **2 m**.
- `coverage` chỉ tính những xe detector có sẵn đã phát hiện, nên xe detector bỏ sót không trừ điểm của bạn.
- RMSE thấp nhưng nhiều ghost, hoặc bỏ rơi nhiều detection, vẫn bị trừ qua `precision_track` và `coverage`.
- Fused **không bắt buộc** tốt hơn LiDAR; chỉ cần không xấu hơn quá 0.05 m.

**Cổng code E–H.** Giảng viên chạy **bộ test gốc** (bản giống `student/tests/test_kalman.py` và `test_tracking_regressions.py`) trên `workspace/` của bạn. Mỗi Part E, F, G, H chỉ đạt khi **mọi** test của Part đó pass:

```text
điểm tự động B = phần nguyên của (fused + lidar + nhất quán) × số_Part_đạt / 4
```

Sửa hoặc xoá test trong repo của bạn không thay đổi kết quả chấm. Test chạy quá 600 giây làm trượt cả 4 Part. **Giữ nguyên tên và chữ ký hàm trong stub.**

**Thủ công (15 điểm)**

| Nội dung | Điểm |
|---|---|
| Spot-check tính đúng E–H; giải thích EKF và gating | 5 |
| Vòng đời track chỉ do LiDAR quyết định (score, xác nhận, xoá) | 5 |
| Diễn giải RMSE cùng ghost/miss; nêu giới hạn của đo camera mô phỏng từ nhãn | 5 |

### 1.3 C — Báo cáo (25 điểm)

Chấm trên `student/SUBMISSION.md`:

| Mức | Điểm | Mô tả |
|---|---|---|
| Xuất sắc | 22–25 | Số liệu khớp `metrics.json`; trả lời đủ 6 câu E–H, đúng và cụ thể, dẫn tới log hoặc code; giải thích được khác biệt hai mode bằng RMSE + matches + ghost/miss; khai báo AI trung thực |
| Đạt | 15–21 | Đủ câu trả lời nhưng một số câu chung chung hoặc thiếu dẫn chứng |
| Yếu | 6–14 | Thiếu nhiều câu, hoặc số liệu không khớp artifacts |
| Không đạt | 0–5 | Không có báo cáo hoặc chỉ chép lại đề |

---

## 2. Bonus (tối đa +5)

Tổng bonus tối đa +5, giảng viên chấm theo chất lượng và bằng chứng. Gợi ý:

| Nội dung | Bằng chứng |
|---|---|
| Export track sang CVAT (`fusion_lab.export_cvat`) và kiểm tra trực quan | File export + ảnh chụp màn hình, mô tả trong `SUBMISSION.md` |
| Trực quan hoá track/đo trên BEV hoặc ảnh camera | Ảnh nhỏ (PNG/JPG) trong `student/artifacts/`, mô tả trong `SUBMISSION.md` |
| Phân tích calibration: làm lệch extrinsic camera, đo ảnh hưởng lên innovation/RMSE | Bảng số liệu + nhận xét trong `SUBMISSION.md` |

`fusion-run-lab` luôn ghi vào `student/artifacts/`. Nếu chạy thử nghiệm bonus (ví dụ lệch calibration), lưu số liệu cần dùng rồi **chạy lại lần chấm điểm** (`--fusion compare --seed 0`, code gốc) trước khi commit artifacts.

---

## 3. Trừ điểm và mất điểm

| Vi phạm | Hậu quả |
|---|---|
| Log thiếu, trùng record, có số không hữu hạn hoặc âm, sai invariant, detection khác nhau giữa hai mode, hoặc tổng không khớp `metrics.json` | **0 điểm tự động** (A và phần tự động của B); phải chạy lại |
| Không chạy `--fusion compare` | Không đạt hạng mục A; mode không có trong log được 0 điểm tracking |
| Sai tên file hoặc cấu trúc thư mục khiến không chấm tự động được (`student/SUBMISSION.md`, `student/artifacts/`) | −5 điểm |
| Không khai báo sử dụng AI trong `SUBMISSION.md` | −10 điểm |
| Commit dữ liệu Waymo (`.tfrecord`), weights (`.pth`), `paths.yaml`, file nén hoặc file > 20 MB | −5 điểm, và phải xoá khỏi lịch sử git |
| Lộ API key hoặc token trong repo | −10 điểm, và phải thu hồi key ngay |
| Nộp muộn | Theo [RULES.md](RULES.md) mục 4 |
| Sao chép bài người khác, hoặc sửa tay / làm giả log và số liệu | **0 điểm toàn bài**, xử lý theo [RULES.md](RULES.md) mục 3 |

---

## 4. Vấn đáp xác minh

Giảng viên có thể gọi **bất kỳ học viên nào** vấn đáp 5 phút, trong buổi lab hoặc
trong vòng 1 tuần sau deadline. Các trường hợp sau chắc chắn bị gọi:

- Lịch sử commit chỉ có 1–2 commit dồn cuối giờ, không theo checkpoint.
- Code hoặc số liệu giống bất thường với bài khác.
- Khai báo AI trong `SUBMISSION.md` không khớp với code đã nộp.
- Số liệu trong `metrics.json` tốt bất thường so với code.

Log và metrics là bằng chứng tự báo cáo; giảng viên có thể chạy lại bài của bạn
với cùng segment, khoảng frame và seed. Trong buổi vấn đáp, bạn phải giải thích
được code và các con số đã nộp. **Phần nào không giải thích được thì phần đó bị tính 0 điểm.**
