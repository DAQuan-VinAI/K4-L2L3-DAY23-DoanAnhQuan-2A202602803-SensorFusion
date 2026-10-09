# Quy định bài lab

Vi phạm các quy định dưới đây bị trừ điểm theo [RUBRIC.md](RUBRIC.md) mục 3.

## 1. Làm bài

- Bài làm **cá nhân**; mỗi học viên tự nộp repo của mình, đặt tên theo [SUBMISSION.md](SUBMISSION.md) mục 2. Được trao đổi ý tưởng, công thức, cách debug với bạn cùng lớp; **không** chia sẻ hoặc chép code, log, số liệu.
- Chỉ viết code trong các hàm Part E–H của `student/workspace/`. **Giữ nguyên tên và chữ ký hàm**: bài được chấm bằng bộ test gốc của giảng viên.
- Không sửa `platform/` để "làm đẹp" kết quả. Nếu bạn nghĩ platform có lỗi, báo lab coach.
- Commit theo checkpoint (`CPx: ...`) như [CHECKPOINTS.md](CHECKPOINTS.md).

## 2. Sử dụng AI

Được phép dùng ChatGPT, Copilot, Claude và các công cụ tương tự, với ba điều kiện:

1. **Khai báo** trong mục "Khai báo sử dụng AI" của `student/SUBMISSION.md`: công cụ, dùng cho phần nào, bạn đã kiểm tra lại thế nào. Không dùng thì ghi "Không dùng AI".
2. **Chịu trách nhiệm**: bạn phải giải thích được mọi dòng code và mọi câu trả lời đã nộp. Phần không giải thích được trong vấn đáp bị tính 0 điểm.
3. **Không bịa**: số liệu trong báo cáo phải lấy từ `metrics.json` và `grade_run.log` của chính bạn. Không nhờ AI "viết hộ" số liệu hoặc nhận xét về kết quả mà bạn chưa chạy.

## 3. Sao chép và gian lận

Các trường hợp sau bị **0 điểm toàn bài** và báo cáo lên chương trình:

- Nộp code, log hoặc báo cáo của người khác (kể cả đã đổi tên biến), hoặc đưa bài của mình cho người khác chép.
- Sửa tay, ghép hoặc tạo giả `metrics*.json`, `grade_run*.log`.
- Sửa code để nhận dạng và "qua mặt" test thay vì cài đúng thuật toán.

Cả người cho chép và người chép đều bị xử lý như nhau.

## 4. Nộp muộn

- **Deadline mặc định: 23:59 ngày học lab, giờ Việt Nam (UTC+7).** Nếu deadline khác, key coach thông báo trong vòng 48 giờ sau buổi lab; khi đó theo thông báo.
- Thời điểm nộp tính theo **commit cuối cùng trên repo nộp** và thời điểm nộp trên LMS (lấy thời điểm muộn hơn).
- Nộp muộn: **−10 điểm**. Mức xử lý cho bài muộn nhiều ngày theo thông báo của giảng viên trên LMS.
- Có lý do chính đáng (ốm, sự cố): báo giảng viên **trước** deadline.

## 5. Sửa bài sau deadline

- Giảng viên chấm **commit cuối cùng trước deadline** (hoặc commit hash bạn nộp trên LMS, nếu sớm hơn).
- Commit sau deadline không được chấm, trừ khi bạn nộp muộn theo mục 4.
- **Không** `git push --force`, rebase hay sửa lịch sử commit sau khi nộp. Lịch sử bị viết lại được xử lý như nộp muộn hoặc không hợp lệ.

## 6. Dữ liệu Waymo và bảo mật

- Dữ liệu Waymo Open Dataset chỉ dùng cho mục đích học tập, theo điều khoản bạn đã chấp nhận khi đăng ký. **Không** chia sẻ lại hoặc commit file `.tfrecord`, ảnh trích từ dataset với số lượng lớn, hoặc link tải riêng.
- **Không** commit weights (`.pth`), `student/config/paths.yaml`, `.env`, file nén hay file > 20 MB.
- **Không** commit API key, token, mật khẩu. Lỡ commit thì **thu hồi key ngay** rồi mới xoá khỏi repo — xoá file không làm key hết hiệu lực.
- Chạy `python tools/check_submission.py` trước khi nộp để phát hiện các lỗi trên.

## 7. Tài nguyên chung

- Tải dữ liệu và weights **ở nhà** (CP0). Không tải dữ liệu lớn qua mạng lớp trong giờ lab.
- Không chia sẻ tài khoản Waymo hoặc tài khoản GitHub.
- Khi kẹt quá 10 phút, hỏi lab coach thay vì xin code của bạn khác.
