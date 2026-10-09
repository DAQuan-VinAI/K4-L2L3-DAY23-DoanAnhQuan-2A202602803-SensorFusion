# Third-party notices

## Vendored or adapted in this repository

- **Simple Waymo Open Dataset Reader** — Grégoire Payen de La Garanderie (gdlg),
  Durham University, Copyright (c) 2019. Apache License 2.0; see
  [the retained LICENSE](platform/third_party/waymo_reader/LICENSE) and
  [upstream repository](https://github.com/gdlg/simple-waymo-open-dataset-reader).
  Its proto schemas retain their Waymo copyright notices. The five range-image
  projection helpers in `utils.py` were restored verbatim from upstream; the
  protobuf Python modules were regenerated for protobuf 6.x. The lab adapter
  exposes vehicle-frame XYZ and range-image intensity.
- **SFA3D FPN-ResNet** — Copyright (c) 2020 Nguyen Mau Dung. MIT License; see
  [the full license](platform/third_party/objdet_models/resnet/LICENSE) and
  [upstream repository](https://github.com/maudzung/SFA3D).
  This attribution covers the vendored network and decoding utilities under
  `platform/third_party/objdet_models/resnet/`, plus the SFA3D-derived BEV
  rasterization and detector adapter in `student/workspace/bev_mapping.py` and
  `student/workspace/detection_pipeline.py`. Model weights are downloaded
  separately and are not included here.

## Data and terms (not shipped in git)

- **Waymo Open Dataset** — data is not included in this repository. Access and
  redistribution are governed by [Waymo's terms](https://waymo.com/open/terms/).
  The course copy is distributed via the class Google Drive; see
  [data acquisition](data/README.md).

## Acknowledged upstream (not vendored here)

- **Complex-YOLOv4-PyTorch** — Copyright Nguyen Mau Dung et al. GNU General
  Public License v3; see
  [upstream repository](https://github.com/maudzung/Complex-YOLOv4-Pytorch).
  The Udacity fusion starter supports this detector alongside SFA3D; this VinUni
  lab uses **FPN-ResNet (SFA3D) only**. No Complex-YOLO source or weights are
  included in this repository.
- **SDCND: Sensor Fusion and Tracking** — Udacity Self-Driving Car Engineer
  Nanodegree Program, Course 2 project starter. Copyright © 2012–2021 Udacity,
  Inc.; educational materials are under Creative Commons
  Attribution-NonCommercial-NoDerivatives 4.0; see
  [upstream LICENSE.md](https://github.com/udacity/nd013-c2-fusion-starter/blob/main/LICENSE.md)
  and [upstream repository](https://github.com/udacity/nd013-c2-fusion-starter).
  This lab follows similar learning goals (LiDAR detection, EKF tracking, camera
  fusion, Waymo sequences) but does **not** redistribute Udacity starter source,
  images, or pre-computed result bundles. Lab instructions, platform runtime,
  tests, and diagrams in this repository are original VinUni course materials.

---

Tracking implementations use standard Kalman-filter and projective-geometry
equations. For a student-facing summary, see [README.md §10 Acknowledgement](README.md#10-acknowledgement).
