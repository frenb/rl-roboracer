#!/usr/bin/env bash
# Regenerate the fly_brain gRPC stubs. Run inside the fly-brain container,
# which pins the only toolchain whose output the Python 3.8 trainer can load
# (see the Dockerfile's grpcio-tools note):
#
#   docker compose exec -T fly-brain bash /fly_brain/gen_protos.sh
#
# Then copy the tree to the trainer so both ends share one definition:
#
#   Copy-Item docker/fly_brain/gen/fly_brain/* rl_agent/fly_brain/ -Recurse -Force
#
# -I is /protos rather than the proto's own directory so the generated import
# is "from fly_brain.proto import fly_brain_pb2", matching virtual_endpoint.
set -euo pipefail

OUT=/fly_brain/gen
rm -rf "$OUT"
mkdir -p "$OUT"

python -m grpc_tools.protoc \
    -I /protos \
    --python_out="$OUT" \
    --grpc_python_out="$OUT" \
    /protos/fly_brain/proto/fly_brain.proto

touch "$OUT/fly_brain/__init__.py" "$OUT/fly_brain/proto/__init__.py"

echo "generated under $OUT:"
find "$OUT" -name '*.py' | sort
