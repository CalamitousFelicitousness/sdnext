#!/usr/bin/env bash
# HunyuanImage-3 Instruct smoke on a fresh CUDA pod (H100/H200 class, 100GB+ disk).
# Usage:
#   git clone --depth 1 -b test/hyimage-h200 https://github.com/CalamitousFelicitousness/sdnext /workspace/sdnext
#   cd /workspace/sdnext && bash test/h200-hyimage-smoke.sh
# Downloads the model, installs sdnext on first launch, generates one image, saves it as smoke.png.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_REPO="CalamitousFelicitousness/HunyuanImage-3.0-Instruct-Distil-SDNQ-4bit-dynamic"
PORT="${SMOKE_PORT:-7860}"
URL="http://127.0.0.1:$PORT"
PAYLOAD="$REPO_DIR/test/h200-hyimage-payload.json"

cd "$REPO_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# model into sdnext's diffusers dir (hf cache layout, resumable, ~47GB)
pip3 install -q -U "huggingface_hub[cli,hf_transfer]"
HF_HUB_ENABLE_HF_TRANSFER=1 hf download "$MODEL_REPO" --cache-dir "$REPO_DIR/models/Diffusers"

# mirrored local settings, with offload set to none for resident weights
[ -f config.json ] || cp test/h200-hyimage-config.json config.json

# SDNQ triton search space, matching the local webui-user.sh
export SDNQ_TRITON_MM_NUM_WARPS_LIST=8
export SDNQ_TRITON_MM_BLOCK_SIZE_N_LIST=64,128
export RUNAI_STREAMER_MEMORY_LIMIT=0

# first launch creates the venv and installs dependencies (10-20 min)
nohup ./webui.sh --debug --listen --port "$PORT" > "$REPO_DIR/smoke-server.log" 2>&1 &
echo "server starting, follow $REPO_DIR/smoke-server.log for install and load progress"
for _ in $(seq 1 240); do
    curl -sf -o /dev/null "$URL/sdapi/v1/sd-models" && break
    sleep 10
done
curl -sf -o /dev/null "$URL/sdapi/v1/sd-models" || { echo "server did not come up, see smoke-server.log"; exit 1; }
echo "server up, submitting generation (model load + think stage take several minutes)"

python3 - "$URL" "$PAYLOAD" "$MODEL_REPO" <<'EOF'
import base64
import json
import sys
import urllib.request

url, payload_path, model_repo = sys.argv[1:4]
models = json.load(urllib.request.urlopen(f"{url}/sdapi/v1/sd-models"))
name = model_repo.split("/")[1]
titles = [m["title"] for m in models if name in m["title"]]
assert titles, f"{name} not found in the model list"
payload = json.load(open(payload_path))
payload.setdefault("override_settings", {})["sd_model_checkpoint"] = titles[0]
print("using checkpoint:", titles[0])
req = urllib.request.Request(f"{url}/sdapi/v1/txt2img", json.dumps(payload).encode(), {"Content-Type": "application/json"})
resp = json.load(urllib.request.urlopen(req, timeout=7200))
images = resp.get("images") or []
assert images, f"no images in response: {str(resp.get('info', ''))[:500]}"
with open("smoke.png", "wb") as f:
    f.write(base64.b64decode(images[0].split(",")[-1]))
print(f"saved smoke.png ({len(images)} image(s)); server also saved to its outputs dir")
print("info:", str(resp.get("info", ""))[:300])
EOF
