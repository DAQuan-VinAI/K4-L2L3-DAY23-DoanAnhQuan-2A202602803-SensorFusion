# Third-party notices

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
- **Waymo Open Dataset** — data is not included in this repository. Access and
  redistribution are governed by [Waymo's terms](https://waymo.com/open/terms/).
  Registration and acceptance are required before receiving the course copy;
  see [data acquisition](data/README.md).

The repository contains no code or images from the Udacity sensor-fusion
starter. Lab instructions and the Mermaid data-flow diagrams are original;
tracking models use standard Kalman-filter and projective-geometry equations.
Third-party detector and range-image projection components are attributed above.
