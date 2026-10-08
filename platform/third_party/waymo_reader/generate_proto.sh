#!/bin/sh
set -eu
cd "$(dirname "$0")"
python -m grpc_tools.protoc -I=. --python_out=. \
    simple_waymo_open_dataset_reader/label.proto \
    simple_waymo_open_dataset_reader/dataset.proto
