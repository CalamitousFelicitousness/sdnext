# HunyuanImage 3.0 Instruct: H200 edit-path findings

Archived 2026-08-20. State of the instruct edit path when the investigation closed. The slice branch `feat/hyimage-instruct` carries the shipped code; this branch adds the H200 reproduction kit on top.

## What works

- txt2img on both SDNQ 4-bit quants: Distil (8 steps, meanflow, cfg factor 1) and full Instruct (50 steps, real cfg pair)
- reference ingestion: the think stage describes every reference exactly, at any input resolution, so the vit path is healthy end to end
- single-reference edits reproduce the reference's layout, wardrobe, pose class and framing
- auto sizing: with width and height unset the model emits its own ratio token and the align-back postprocess restores the input aspect ratio

## What falls short

- identity: faces regenerate as a similar person rather than the same person, even on single-reference edits
- multi-reference edits re-render Image 1 regardless of instruction targeting; this matches the upstream fusion convention where Image 1 is the base and later images contribute attributes
- output from the full-Instruct quant carries high-frequency noise; that repo still has its 32 MoE router gates quantized (8 to 10 percent weight error), while the Distil repo had them restored
- fine instruction detail lands weakly at the default guidance of 2.5

## Open probes

- Distil single-reference A/B as the noise discriminator (routers fixed there)
- guidance above 2.5 on the full model
- router unquantization for the full Instruct repo (costs a roughly 50GB download)
- bf16 weights on a B200 (192GB) to separate the model's ceiling from quant damage
- `drop_think` wiring, stripping the think text from the diffusion prefill while keeping the recaption
- base-image-first reference ordering for multi-reference edits

## Rig

`test/modal_sdnext_smoke.py` and the nightly variant run the whole loop from a workstation: baked dependency image, shared weights volume, branch tip fetched per launch with the revision printed at startup. `test/h200-hyimage-smoke.sh` is the raw-pod equivalent.

```bash
modal run test/modal_sdnext_smoke.py --image base.png --image2 source.png --prompt "..." \
  [--model <hf-repo>] [--steps N] [--cfg X] [--no-think]
```

Constraints that survive any environment: weights resident (balanced offload flips the device mid staged-generate, group offload breaks the SigLIP path), stock sdpa, SDNQ quantized matmul disabled.
