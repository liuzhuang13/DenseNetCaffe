set -e

TOOLS=/usr/local/caffe/build/tools

$TOOLS/caffe train \
  --solver=solver.prototxt