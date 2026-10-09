# Dữ liệu và weights — Day 23

## Tải dữ liệu Waymo

Giảng viên đã tải sẵn 4 segments của khóa học. Tải từ Google Drive:
[Day23-Fusion](https://drive.google.com/drive/folders/1XTXA60c9gV6rkK6fWohMyHw8qpsfYBlE?usp=sharing) (thư mục `Waymo/` và `weights/`).

Dữ liệu Waymo Open Dataset chỉ dùng cho học tập trong lab; không đưa lên repo công khai
và không chia sẻ lại ra ngoài lớp.

Danh sách segment của khóa học:

1. `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord` (mặc định).
2. `training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord`
3. `training_segment-10094743350625019937_3420_000_3440_000_with_camera_labels.tfrecord`
4. `training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord`

Đặt các file vào `data/Waymo/`.

## Weights SFA3D

Lấy `fpn_resnet_18_epoch_300.pth` trong thư mục `weights/` của
[Google Drive Day23-Fusion](https://drive.google.com/drive/folders/1XTXA60c9gV6rkK6fWohMyHw8qpsfYBlE?usp=sharing), hoặc tải từ
[checkpoints/fpn_resnet_18 của SFA3D](https://github.com/maudzung/SFA3D/tree/master/checkpoints/fpn_resnet_18)
(MIT; tác giả Nguyen Mau Dung). Đặt đúng đường dẫn:

```text
data/weights/pretrained_fpn-resnet/fpn_resnet_18_epoch_300.pth
```

Đường dẫn này được `fusion_lab.scripts.run_lab._resolve_weights` tìm khi
`weights_dir` trỏ tới `data/weights`. Không commit `.tfrecord`, `.pth`, hay bản sao
dữ liệu. [Cấu hình mẫu](../student/config/paths.example.yaml) dùng `../data/Waymo`
và `../data/weights`, tính từ thư mục `student/`; runtime giải quyết các đường dẫn
đó ngay cả khi lệnh chạy từ root repo.
