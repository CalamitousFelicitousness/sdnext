"""Run an sdnext smoke on a Modal GPU without touching a pod.

One-time setup:
    pip install modal
    modal setup

Usage:
    modal run modal_sdnext_smoke.py                          # generate, save smoke.png locally
    modal run modal_sdnext_smoke.py --payload my.json        # custom generation payload
    modal run modal_sdnext_smoke.py --out result.png
    modal run modal_sdnext_smoke.py::ui                      # tunnel the web UI for interactive use
    modal run modal_sdnext_smoke.py --image a.png --image2 b.png --prompt "..."   # instruct edit with reference images

The image bakes sdnext (branch below) with dependencies installed; the model and
server-side outputs persist on the 'sdnext-models' volume, so only the first run
downloads weights. The branch tip is re-fetched at every run, so branch churn does
not require an image rebuild. GPU and branch are set below.
"""

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request

import modal

GPU = os.environ.get("SMOKE_GPU", "H200")
REPO_URL = "https://github.com/CalamitousFelicitousness/sdnext"
BRANCH = "test/hyimage-h200"
MODEL_REPO = "CalamitousFelicitousness/HunyuanImage-3.0-Instruct-Distil-SDNQ-4bit-dynamic"
REPO_DIR = "/sdnext"
PORT = 7860
URL = f"http://127.0.0.1:{PORT}"

app = modal.App("sdnext-smoke")
vol = modal.Volume.from_name("sdnext-models", create_if_missing=True)
# env vars (e.g. HF_TOKEN) live in the workspace secret, not in this script:
#   modal secret create huggingface HF_TOKEN=hf_...
secrets = [modal.Secret.from_name("huggingface")]

image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install("git", "curl", "libgl1", "libglib2.0-0", "google-perftools")
    .pip_install("uv", "setuptools", "wheel")
    # no venv in the container: without this, every 'uv pip' in sdnext's installer
    # refuses the system interpreter and falls back to pip
    .env({"UV_SYSTEM_PYTHON": "1"})
    .run_commands(f"git clone --depth 1 -b {BRANCH} {REPO_URL} {REPO_DIR}")
    # bake dependencies: the installer needs a GPU visible to pick CUDA torch;
    # the trailing test generation has no model and may fail, deps are in by then
    .run_commands(f"cd {REPO_DIR} && python launch.py --debug --test --use-cuda --uv || true", gpu="T4")
    # the || true above tolerates the test failure but must not mask a failed bake
    .run_commands("python -c 'import torch, diffusers, transformers; print(torch.__version__)'")
)


def launch_server(mirror: bool = False, offload: str = "none", model_repo: str = MODEL_REPO):
    subprocess.run(["git", "-C", REPO_DIR, "fetch", "--depth", "1", "origin", BRANCH], check=True)
    subprocess.run(["git", "-C", REPO_DIR, "reset", "--hard", "FETCH_HEAD"], check=True)
    # print the revision so a run log always identifies the code it tested
    rev = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    print(f"sdnext {BRANCH} @ {rev}")

    from huggingface_hub import snapshot_download
    snapshot_download(model_repo, cache_dir="/models/Diffusers")
    vol.commit()

    cfg = json.load(open(f"{REPO_DIR}/test/h200-hyimage-config.json"))
    # none: fully resident; expandable_segments handles transient headroom.
    # balanced breaks the hyimage staged generate (device mismatch at decode gather)
    cfg["diffusers_offload_mode"] = offload
    cfg["diffusers_dir"] = "/models/Diffusers"
    cfg["outdir_samples"] = "/models/outputs"
    cfg["outdir_grids"] = "/models/outputs"
    # per-type dirs mirror the local layout; sdnext defaults embed an extra outputs/ prefix
    cfg.update({
        "outdir_txt2img_samples": "text",
        "outdir_img2img_samples": "image",
        "outdir_control_samples": "control",
        "outdir_extras_samples": "extras",
        "outdir_init_images": "inputs",
        "outdir_txt2img_grids": "grids",
        "outdir_img2img_grids": "grids",
        "outdir_control_grids": "grids",
        "outdir_save": "save",
        "outdir_video": "video",
    })
    if mirror:
        # full local mirror, reproduces the SDNQ triton autotune address fault
        os.environ["SDNQ_TRITON_MM_NUM_WARPS_LIST"] = "8"
        os.environ["SDNQ_TRITON_MM_BLOCK_SIZE_N_LIST"] = "64,128"
    else:
        # SDNQ attention faults in triton autotune on every tested arch; run stock sdpa.
        # int8 matmul compiles a fullgraph helper per layer geometry and hard-fails
        # dynamo's accumulated recompile limit on MoE models; bf16 matmul is faster here anyway
        cfg["sdp_overrides"] = []
        cfg["sdnq_quantize_matmul_mode"] = "disabled"
    with open(f"{REPO_DIR}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    os.environ["RUNAI_STREAMER_MEMORY_LIMIT"] = "0"

    # autotune benches churn odd-sized transients; without this the allocator cache
    # fragments up to the vram ceiling (native linux; the WSL2 VMM crash does not apply).
    # sdnext sets its allocator conf with setdefault, so this pre-set wins
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.8"

    # persist kernel caches across runs; a fresh container otherwise repays every autotune
    os.makedirs("/models/cache/triton", exist_ok=True)
    os.makedirs("/models/cache/inductor", exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = "/models/cache/triton"
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = "/models/cache/inductor"

    proc = subprocess.Popen(
        [sys.executable, "launch.py", "--debug", "--listen", "--port", str(PORT), "--uv"],
        cwd=REPO_DIR,
    )
    for _ in range(180):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited during startup: {proc.returncode}")
        try:
            urllib.request.urlopen(f"{URL}/sdapi/v1/sd-models", timeout=5)
            return proc
        except Exception:
            time.sleep(5)
    raise RuntimeError("server did not come up in 15 minutes")


def monitor_memory(interval: float):
    while True:
        time.sleep(interval)
        try:
            m = json.load(urllib.request.urlopen(f"{URL}/sdapi/v1/memory", timeout=5))
            cuda = m.get("cuda", {})
            gb = 1 << 30
            sysm, act, res, ev = (cuda.get(k, {}) for k in ("system", "active", "reserved", "events"))
            print(
                f"[mem] gpu={sysm.get('used', 0) / gb:.1f}/{sysm.get('total', 0) / gb:.1f}GB"
                f" active={act.get('current', 0) / gb:.1f}GB peak={act.get('peak', 0) / gb:.1f}GB"
                f" reserved={res.get('current', 0) / gb:.1f}GB"
                f" retries={ev.get('retries', 0)} oom={ev.get('oom', 0)}"
            )
        except Exception:
            pass


def resolve_checkpoint(model_repo: str = MODEL_REPO) -> str:
    models = json.load(urllib.request.urlopen(f"{URL}/sdapi/v1/sd-models"))
    name = model_repo.split("/")[1]
    titles = [m["title"] for m in models if name in m["title"]]
    if not titles:
        raise RuntimeError(f"{name} not in model list: {[m['title'] for m in models][:20]}")
    return titles[0]


def load_checkpoint(title: str):
    # autoload is off, and generation aborts rather than loading a model itself;
    # setting the option triggers the load synchronously within this request
    req = urllib.request.Request(
        f"{URL}/sdapi/v1/options",
        json.dumps({"sd_model_checkpoint": title}).encode(),
        {"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=3600)
    current = json.load(urllib.request.urlopen(f"{URL}/sdapi/v1/options")).get("sd_model_checkpoint")
    if current != title:
        raise RuntimeError(f"checkpoint did not stick: wanted {title}, got {current}")


@app.function(image=image, gpu=GPU, volumes={"/models": vol}, secrets=secrets, timeout=2 * 3600)
def smoke(payload: dict, mirror: bool = False, monitor: float = 10, offload: str = "none", no_think: bool = False, model: str = "") -> dict:
    if no_think:
        os.environ["SD_HYIMAGE_BOT_TASK"] = "image"  # diffuse from the raw prompt, no think/recaption stage
    launch_server(mirror=mirror, offload=offload, model_repo=model or MODEL_REPO)
    if monitor > 0:
        threading.Thread(target=monitor_memory, args=(monitor,), daemon=True).start()
    try:
        title = resolve_checkpoint(model or MODEL_REPO)
        print("loading checkpoint:", title)
        load_checkpoint(title)
        payload.setdefault("override_settings", {})["sd_model_checkpoint"] = title
        endpoint = "img2img" if payload.get("init_images") else "txt2img"
        req = urllib.request.Request(
            f"{URL}/sdapi/v1/{endpoint}",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        resp = json.load(urllib.request.urlopen(req, timeout=7000))
        if not resp.get("images"):
            raise RuntimeError(f"no images in response: {str(resp.get('info', ''))[:500]}")
        return {"images": resp["images"], "info": str(resp.get("info", ""))[:1000]}
    finally:
        vol.commit()  # persist kernel caches and server-side saves even when the run fails


@app.function(image=image, gpu=GPU, volumes={"/models": vol}, secrets=secrets, timeout=4 * 3600)
def ui(mirror: bool = False, monitor: float = 10, offload: str = "none", no_think: bool = False, model: str = ""):
    if no_think:
        os.environ["SD_HYIMAGE_BOT_TASK"] = "image"
    with modal.forward(PORT) as tunnel:
        proc = launch_server(mirror=mirror, offload=offload, model_repo=model or MODEL_REPO)
        if monitor > 0:
            threading.Thread(target=monitor_memory, args=(monitor,), daemon=True).start()
        print(f"sdnext UI: {tunnel.url}")
        print("runs until timeout or ctrl-c on the modal run")
        proc.wait()


@app.local_entrypoint()
def main(payload: str = "", out: str = "smoke.png", mirror: bool = False, monitor: float = 10, offload: str = "none", image: str = "", image2: str = "", prompt: str = "", scale: float = 0, no_think: bool = False, model: str = "", steps: int = 0, cfg: float = 0):
    import base64
    import pathlib

    if payload:
        data = json.load(open(payload))
    elif image and not prompt:
        raise SystemExit("edit mode needs --prompt describing the transfer, or a full --payload")
    else:
        data = {
            "prompt": "a lighthouse on a rocky coastline at golden hour, crashing waves, dramatic clouds, photorealistic",
            "negative_prompt": "",
            "steps": 8,
            "cfg_scale": 2.5,
            "width": 1024,
            "height": 1024,
            "seed": 42,
            "batch_size": 1,
            "save_images": True,
        }
    if prompt:
        data["prompt"] = prompt
    if steps:
        data["steps"] = steps
    if cfg:
        data["cfg_scale"] = cfg
    ref_paths = [p for p in (image, image2) if p]
    if ref_paths:
        refs = []
        for p in ref_paths:
            raw = pathlib.Path(p).read_bytes()
            if scale > 0:
                # references enter the sequence as vit tokens at their own resolution and the
                # output size follows the input, so scaling must happen before upload;
                # the server-side scale_by option only shrinks the generated image
                import io
                from PIL import Image
                im = Image.open(io.BytesIO(raw))
                im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                raw = buf.getvalue()
            refs.append(base64.b64encode(raw).decode())
        data["init_images"] = refs
    result = smoke.remote(data, mirror=mirror, monitor=monitor, offload=offload, no_think=no_think, model=model)
    img = base64.b64decode(result["images"][0].split(",")[-1])
    pathlib.Path(out).write_bytes(img)
    print(f"saved {out} ({len(img)} bytes)")
    print("info:", result["info"][:300])
