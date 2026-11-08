# Round-two organizer reproduction check

Status: **IN PROGRESS**  
Started: 2026-11-08 16:40 UTC (local workspace time)  
Scope: reproduce `adapted_entry_v17g_rules` inside `python:3.13` with the adaptation round-two kit under organizer CPU, memory, network, and output constraints.

## Checklist

- [x] Read entry instructions and reproduction metadata
- [x] Confirm regenerated `gate.json` says `kit_round: adaptation`
- [x] Create workspace-only git archive copy and overlay gate/bundled assets
- [ ] Install pinned requirements in `python:3.13` with network available (image pull/build running)
- [ ] Run offline with full kit mounted read-only
- [ ] Run offline with only train/test mounted read-only
- [ ] Run checker on both outputs
- [ ] Compare both outputs and reference submission (if present)

## Initial inspection

- Entry README command: `python predict.py --data /data --out /output/submission.csv`
- Image: `python` (organizer says Python 3.13)
- Pinned requirements: `numpy==2.3.4`, `scipy==1.16.3`
- Docker client/server: 29.5.3 / 29.5.3
- Gate status and all run results pending.


## Entry snapshot and kit

- Gate condition was satisfied before copying: `kit_round` is `adaptation`.
- Archive source was `R2`, selector `HEAD adapted_entry_v17g_rules`; extracted under `run/entry/adapted_entry_v17g_rules/`.
- Regenerated `gate.json` and the source entry's complete `bundled/` directory were overlaid after archive extraction.
- Kit metadata: `round=adaptation`, `max_points=20`; `train/`, `test/`, and `validation/` are present.
- Snapshot SHA-256 values are recorded below from source and workspace copy.
- Docker image build and both offline runs pending.

| File | SHA-256 |
|---|---|
| `R2/adapted_entry_v17g_rules/gate.json` | `9e71c9662ffd9a4654444dc9623f19c91c78aee00fa1cff426af7c0da8d4ce29` |
| `R2/adapted_entry_v17g_rules/predict.py` | `68a80a91cbbb580c7544b82237d210bf4a392c2f3c4729648393a41e9baefb1f` |
| `R2/adapted_entry_v17g_rules/reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` |
| `R2/adapted_entry_v17g_rules/requirements.txt` | `988fcfe105d1a1a4b59d28e778ec402015fdab4b2b15f040800315239d77b2ab` |
| `run/entry/adapted_entry_v17g_rules/gate.json` | `9e71c9662ffd9a4654444dc9623f19c91c78aee00fa1cff426af7c0da8d4ce29` |
| `run/entry/adapted_entry_v17g_rules/predict.py` | `68a80a91cbbb580c7544b82237d210bf4a392c2f3c4729648393a41e9baefb1f` |
| `run/entry/adapted_entry_v17g_rules/reproduce.json` | `cb91250a431dc92a45ceda7bf66df8006a4ecbce2b8e393f52f7284fd22246a1` |
| `run/entry/adapted_entry_v17g_rules/requirements.txt` | `988fcfe105d1a1a4b59d28e778ec402015fdab4b2b15f040800315239d77b2ab` |

## Docker host facts

- Docker Desktop VM: 11 CPUs, 8,321,372,160 bytes RAM (~7.75 GiB), `aarch64` Linux.
- Container requests will use 2 CPUs and 16 GiB as organizer-specified. The Docker VM itself has less than 16 GiB total, so this host cannot provide the full requested memory even though the per-container cap is set to 16 GiB.
- `python:3.13` resolves to an ARM64 image in this environment; the organizer architecture was not supplied for comparison.


## Image preparation

- Docker build installed the pinned requirements successfully from `requirements.txt`: numpy 2.3.4 and scipy 1.16.3; Python base resolved as `linux/arm64`.
- Image: `adaptation-entry-check:local` (`sha256:8684bd330e4419c2e257bb673e4d181f117419f5f5869940c28c8b77054da14f`).
- Build elapsed: 89.71 seconds (`/usr/bin/time -p`), including image pull and pip install.
- Initial shell wrapper printed `zsh:33: read-only variable: status` after the successful build because `status` is reserved in zsh; image inspection confirmed the build succeeded. Runtime wrapper uses `rc` instead.
- Full-kit offline run is now in progress. Its root filesystem and `/app` are read-only; `/tmp` and the dedicated `/output` bind mount are writable.


### Full-kit run progress

- Full kit is mounted read-only at `/data`; the dedicated fresh `run/output-full` directory is writable at `/output`.
- Verified container settings: 2 CPUs, 17,179,869,184-byte memory cap, `NetworkMode=none`, read-only root filesystem, and mount flags `/data RW=false`, `/output RW=true` (see `run/logs/full-inspect.txt`).
- Python 3.13.18, numpy 2.3.4, scipy 1.16.3, ARM64 Linux (see `run/logs/runtime-versions.txt`).
- At 16:45 UTC, sysid completed its 1,536-row prediction; v13 and exact members were still running. Sampled container memory: 336 MiB used of Docker VM's 7.75 GiB; CPU around 200%.

