# UMD MindLabs GPU Cluster — Server Guide
  
Complete reference for the MindLabs GPU cluster at the University of Maryland. Covers hardware, software, services, access methods, and operational procedures for the Agent-Graph-Memory project.

---

## Table of Contents

1. [Cluster Overview](#1-cluster-overview)
2. [Hardware Specs](#2-hardware-specs)
3. [Software Environment](#3-software-environment)
4. [Storage Topology](#4-storage-topology)
5. [Running Services](#5-running-services) — includes [Using vLLM](#using-vllm-openai-compatible-client)
6. [Access Methods](#6-access-methods)
7. [Operational Procedures](#7-operational-procedures)
8. [Hard Rules & Gotchas](#8-hard-rules--gotchas)
9. [Project Integration](#9-project-integration)
10. [Quick Reference](#10-quick-reference)

---

## 1. Cluster Overview

```
┌─────────────────────────────────────────────────────────┐
│                 Local Dev Machine (laptop)                │
│                                                           │
│  SSH tunnels:                                             │
│  -L 8000:localhost:8000     → gpu02 (vLLM)               │
│  -L 11435:localhost:11434   → gpu01 (Ollama)             │
└────────────────────────┬────────────────────────────────┘
                         │
                         │ SSH (jump host)
                         ▼
┌─────────────────────────────────────────────────────────┐
│           mind-access00.cs.umd.edu (Jump Node)            │
│  - Login server, no GPUs                                  │
│  - Gateway to internal GPU nodes                          │
└──────┬──────────────────────────────────┬───────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────────────┐     ┌──────────────────────┐
│  gpu01.mind.cs.umd.edu │     │  gpu02.mind.cs.umd.edu │
│  (Ollama node)         │     │  (vLLM node)           │
│                        │     │                        │
│  2x NVIDIA L40S        │     │  2x NVIDIA L40S        │
│  Ollama :11434         │     │  vLLM :8000            │
│  24 models installed   │     │  /scratch 5.8TB RAID0  │
│  4x NVMe (unmounted)   │     │  root 89% full!        │
└──────────────────────┘     └──────────────────────┘
       │                                  │
       └──────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│              Shared Storage (GlusterFS + NFS)             │
│  /home/<username> → fs00.mind.local (GlusterFS)          │
│  /fs/scratch     →  fs01.mind.local:/zpool0/scratch (NFS)│
└─────────────────────────────────────────────────────────┘
```

**Node roles:**

| Node | Hostname | Role | Services |
|------|----------|------|----------|
| Jump | `mind-access00.cs.umd.edu` | SSH gateway, no GPUs | — |
| GPU01 | `gpu01.mind.cs.umd.edu` | Embedding server | Ollama (`:11434`) |
| GPU02 | `gpu02.mind.cs.umd.edu` | LLM inference server | vLLM (`:8000`) |

---

## 2. Hardware Specs

### GPU Nodes (gpu01 & gpu02 — same CPU/GPU/RAM, different NVMe layout)

| Component | Spec |
|-----------|------|
| GPU | 2x NVIDIA L40S (46,068 MiB VRAM each, ~92 GB total) |
| GPU Driver | 590.48.01 |
| CUDA Version | 13.1 |
| CPU | 2x Intel Xeon 6515P (32 cores, 64 threads total) |
| RAM | 251 GB |
| OS | Rocky Linux 9.7 (Blue Onyx) |
| Boot Disk | 447 GB NVMe (LVM: 70 GB root, 4 GB swap, ~370 GB home) |
| Python | 3.13.13 (miniconda3 base env) |
| Conda | `~/miniconda3` (base env active; per-user home) |

**NVMe layout differences:**

| | gpu01 | gpu02 |
|--|-------|-------|
| Boot NVMe | 447 GB (LVM) | 447 GB (LVM) |
| 4x 1.5 TB NVMe | **Not mounted** (available for manual use) | **RAID0 → 5.8 TB `/scratch`** (mounted) |
| NFS `/fs/scratch` | Mounted (shared) | **Not mounted** |
| Root disk usage | 31% (49 GB free) | **89% (8 GB free)** |

> **Warning:** gpu02 root disk is at 89% capacity (only 8 GB free). Avoid writing large files to `/` on gpu02. Use `/scratch` (5.8 TB local RAID0) or `/fs/scratch` (NFS, if you mount it) instead.

### Jump Node (mind-access00)

| Component | Spec |
|-----------|------|
| OS | Rocky Linux 8.6 (Green Obsidian) |
| Kernel | 4.18.0-372.16.1.el8_6.x86_64 |
| CPU | 4x Intel Xeon E5-2690 v4 @ 2.60GHz (4 cores, 1 thread/core) |
| RAM | 7.5 GB |
| Disk | 190 GB root (4% used) |
| GPU | None |
| Role | SSH gateway / login server |
| Storage | `/home` via GlusterFS (auto.home) |

---

## 3. Software Environment

### GPU01 (Ollama node) & GPU02 (vLLM node) — identical packages

Both nodes see the same per-user conda env via GlusterFS (`~/miniconda3`).

| Package | Version |
|---------|---------|
| Python | 3.13.13 |
| vllm | 0.21.0 |
| torch | 2.11.0 |
| torchaudio | 2.11.0 |
| torchvision | 0.26.0 |
| transformers | 5.8.1 |
| flashinfer-python | 0.6.8.post1 |
| flashinfer-cubin | 0.6.8.post1 |
| torch_c_dlpack_ext | 0.1.5 |

### CUDA / Driver

| Property | Value |
|----------|-------|
| NVIDIA-SMI | 590.48.01 |
| Driver Version | 590.48.01 |
| CUDA Version | 13.1 |

Verified on both gpu01 and gpu02 (identical).

### Conda

Only the `base` environment is typically set up. Install packages in your own conda env at `~/miniconda3`. Since `/home` is shared via GlusterFS, the same env is available on both GPU nodes for your account.

---

## 4. Storage Topology

### Shared Storage (GlusterFS)

| Mount | Source | Type | Notes |
|-------|--------|------|-------|
| `/home/<username>` | `fs00.mind.local:/home/<username>` | fuse.glusterfs | Shared across all nodes. Conda, code, home dir. |

### Network Scratch (NFS)

| Mount | Source | Type | Notes |
|-------|--------|------|-------|
| `/fs/scratch` | `fs01.mind.local:/zpool0/scratch` | nfs4 | Shared scratch space. Good for large temp files. |

### Local NVMe — gpu01

| Device | Size | Mount | Notes |
|--------|------|-------|-------|
| `nvme1n1` | 447 GB | LVM (root/swap/home) | Boot drive |
| `nvme0n1` | 1.5 TB | not mounted | Available for manual use / TRITON_CACHE_DIR |
| `nvme2n1` | 1.5 TB | not mounted | Available for manual use |
| `nvme3n1` | 1.5 TB | not mounted | Available for manual use |
| `nvme4n1` | 1.5 TB | not mounted | Available for manual use |

### Local NVMe — gpu02

| Device | Size | Mount | Notes |
|--------|------|-------|-------|
| `nvme1n1` | 447 GB | LVM (root/swap/home) | Boot drive |
| `nvme0n1` | 1.5 TB | `md127` (RAID0) | Part of 5.8 TB `/scratch` array |
| `nvme2n1` | 1.5 TB | `md127` (RAID0) | Part of 5.8 TB `/scratch` array |
| `nvme3n1` | 1.5 TB | `md127` (RAID0) | Part of 5.8 TB `/scratch` array |
| `nvme4n1` | 1.5 TB | `md127` (RAID0) | Part of 5.8 TB `/scratch` array |

> **gpu02 `/scratch`** is a 4-disk RAID0 (5.8 TB) mounted at `/scratch`. Fast local storage — ideal for TRITON_CACHE_DIR, model weights, or large temp files. Use `/scratch/triton_cache_gemma` instead of `/tmp/triton_cache_gemma` on gpu02.
>
> **gpu01** has the same 4x 1.5 TB NVMe but they are **not mounted**. You can mount them manually or use `/tmp` (backed by root NVMe) for TRITON_CACHE_DIR.

### The GlusterFS + Triton Problem

GlusterFS causes **Triton JIT race conditions**. When vLLM compiles CUDA kernels via Triton, the cache is written to a shared path. Multiple processes on different nodes reading/writing the same GlusterFS cache directory causes JIT compilation to fail or hang.

**Fix:** Always set `TRITON_CACHE_DIR` to a local NVMe path:
```bash
# On gpu02 (has /scratch RAID0):
export TRITON_CACHE_DIR=/scratch/triton_cache_gemma

# On gpu01 (NVMe not mounted, use /tmp on root NVMe):
export TRITON_CACHE_DIR=/tmp/triton_cache_gemma
```

This is baked into the vLLM launch command (see [Section 7](#7-operational-procedures)).

---

## 5. Running Services

### Ollama (gpu01 :11434)

| Property | Value |
|----------|-------|
| Host | `gpu01.mind.cs.umd.edu` |
| Port | `11434` |
| Status | **Running** (PID 2773, since Jun 25) |
| Process | `/usr/local/bin/ollama serve` |
| API | OpenAI-compatible (`/v1/embeddings`, `/v1/chat/completions`) + native (`/api/tags`, `/api/generate`) |

**Installed models (24 total as of Jun 29, 2026):**

| Model | Params | Quant | Family | Use Case |
|-------|--------|-------|--------|----------|
| `mxbai-embed-large:335m` | 334M | F16 | bert | **Embeddings** (project default) |
| `mxbai-embed-large:latest` | 334M | F16 | bert | Embeddings (alias) |
| `gemma4:31b` | 31.3B | Q4_K_M | gemma4 | Chat (project default for Ollama) |
| `gemma4:26b-a4b-it-qat` | 25.2B | Q4_0 | gemma4 | Chat (lighter gemma4) |
| `gemma4:12b` | 11.9B | Q4_K_M | gemma4 | Chat (fast) |
| `gemma3:27b` | 27.4B | Q4_K_M | gemma3 | Chat (older gen) |
| `gemma3:4b` | 4.3B | Q4_K_M | gemma3 | Chat (very fast) |
| `qwen3:32b` | 32.8B | Q4_K_M | qwen3 | Chat (thinking) |
| `qwen3:8b` | 8.2B | Q4_K_M | qwen3 | Chat (thinking, fast) |
| `qwen3:4b` | 4.0B | Q4_K_M | qwen3 | Chat (thinking, very fast) |
| `deepseek-r1:70b` | 70.6B | Q4_K_M | llama | Chat (reasoning, largest) |
| `deepseek-r1:32b` | 32.8B | Q4_K_M | qwen2 | Chat (reasoning) |
| `deepseek-r1:14b` | 14.8B | Q4_K_M | qwen2 | Chat (reasoning, fast) |
| `mistral:latest` | 7.2B | Q4_0 | llama | Chat (lightweight) |
| `mistral-small3.1:latest` | 24.0B | Q4_K_M | mistral3 | Chat |
| `llama4:latest` | 108.6B | Q4_K_M | llama4 | Chat (largest model) |
| `llama3.2:latest` | 3.2B | Q4_K_M | llama | Chat (very lightweight) |
| `llama3.2-vision:11b` | 9.8B | Q4_K_M | mllama | Vision + chat |
| `gpt-oss:20b` | 20.9B | MXFP4 | gptoss | Chat (OpenAI open-source) |
| `kimi-k2-thinking:cloud` | 1T | INT4 | deepseek2 | Chat (remote/cloud proxy) |
| `qwen2.5vl:32b` | 33.5B | Q4_K_M | qwen25vl | Vision + chat |
| `qwen2.5vl:7b` | 8.3B | Q4_K_M | qwen25vl | Vision + chat (fast) |
| `granite3.2-vision:latest` | 2.5B | Q4_K_M | granite | Vision + chat (lightweight) |
| `granite3.2-vision:2b` | 2.5B | Q4_K_M | granite | Vision + chat (lightweight) |

**Verify Ollama is running:**
```bash
# From within the cluster (jump node or any GPU node):
curl -s gpu01.mind.cs.umd.edu:11434/api/tags | python3 -m json.tool

# From local laptop (requires SSH tunnel, see Section 6):
curl -s localhost:11435/api/tags | python3 -m json.tool
```

**List installed models:**
```bash
curl -s gpu01.mind.cs.umd.edu:11434/api/tags
```

**Pull a new model:**
```bash
ssh gpu01.mind.cs.umd.edu
ollama pull <model_name>
```

### vLLM (gpu02 :8000)

| Property | Value |
|----------|-------|
| Host | `gpu02.mind.cs.umd.edu` |
| Port | `8000` |
| Status | **RUNNING** (as of Jul 7, 2026) |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507` (served name), weights at `/scratch/models/qwen3-30b-a3b-instruct-2507-fp8` |
| Tensor Parallel | 2 (uses both L40S GPUs) |
| Max Model Len | 131072 tokens |
| GPU Memory Util | 0.85 |
| API | OpenAI-compatible (`/v1/chat/completions`, `/v1/models`) |

> **Current state (2026-07-07):** serving **Qwen3-30B-A3B-Instruct-2507-FP8** (MoE,
> 3B active params — several× Nemotron's throughput; non-thinking instruct). Swapped in
> for EXP-MODEL-LOCAL after Phase 4 closed. Nemotron-30B weights remain at
> `/scratch/models/nemotron-nano-30b-fp8`; its exact relaunch command is in §7.1.
> NOTE: `NEMOTRON_THINKING`/budget-ceiling env vars are inert for Qwen3.

**Verify vLLM is running:**
```bash
# From within the cluster:
curl -s gpu02.mind.cs.umd.edu:8000/v1/models | python3 -m json.tool

# From local laptop (requires SSH tunnel):
curl -s localhost:8000/v1/models | python3 -m json.tool
```

**Health check:**
```bash
curl -s gpu02.mind.cs.umd.edu:8000/health
```

### Using vLLM (OpenAI-compatible client)

vLLM on gpu02 exposes an OpenAI-compatible HTTP API. Call it from any cluster node
(or JupyterHub) directly, or from your laptop via the SSH tunnel in §6.2.

| Access from | Base URL |
|-------------|----------|
| Cluster / JupyterHub | `http://gpu02.mind.cs.umd.edu:8000/v1` |
| Laptop (SSH tunnel) | `http://localhost:8000/v1` |

**Served model name** (pass this exact string as `model=`):

```
Qwen/Qwen3-30B-A3B-Instruct-2507
```

> Confirm the live name anytime with `curl -s …/v1/models`. If someone relaunched
> a different checkpoint, use whatever `id` that endpoint returns.

#### curl

```bash
curl -s http://gpu02.mind.cs.umd.edu:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "messages": [
      {"role": "user", "content": "Explain tensor parallelism in one sentence."}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }' | python3 -m json.tool
```

#### Python (`openai` SDK)

```bash
pip install openai   # once, in your conda env
```

```python
from openai import OpenAI

# From a cluster node / JupyterHub:
client = OpenAI(
    base_url="http://gpu02.mind.cs.umd.edu:8000/v1",
    api_key="EMPTY",  # vLLM does not require a real key
)

# From your laptop (after SSH tunnel → localhost:8000):
# client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is a Mixture-of-Experts (MoE) model?"},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(resp.choices[0].message.content)
```

#### Streaming

```python
stream = client.chat.completions.create(
    model="Qwen/Qwen3-30B-A3B-Instruct-2507",
    messages=[{"role": "user", "content": "Count from 1 to 5."}],
    max_tokens=64,
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
print()
```

#### List models / quick sanity check

```python
print(client.models.list())
```

```bash
# Same check without Python:
curl -s http://gpu02.mind.cs.umd.edu:8000/v1/models | python3 -m json.tool
```

---

## 6. Access Methods

### 6.1 SSH (Laptop → Jump Node → GPU Nodes)

**Direct SSH to jump node:**
```bash
ssh <username>@mind-access00.cs.umd.edu
```

**From jump node to GPU nodes (passwordless within cluster):**
```bash
ssh gpu01.mind.cs.umd.edu
ssh gpu02.mind.cs.umd.edu
```

### 6.2 SSH Tunnels (Local Dev Access)

Services on GPU nodes are **not exposed outside the firewall**. To access them from your laptop, create SSH tunnels through the jump node.

**Start both tunnels (run from your laptop):**
```bash
# vLLM tunnel (gpu02 :8000 → localhost:8000)
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 8000:localhost:8000 <username>@gpu02.mind.cs.umd.edu

# Ollama tunnel (gpu01 :11434 → localhost:11435)
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 11435:localhost:11434 <username>@gpu01.mind.cs.umd.edu
```

**Why port 11435 for Ollama?** Local port 11434 may be occupied by a local Ollama install. Using 11435 avoids conflicts. The `.env` file maps `EMBEDDING_BASE_URL=http://localhost:11435/v1` accordingly.

**Verify tunnels are active:**
```bash
lsof -iTCP:8000 -iTCP:11435 -sTCP:LISTEN
```

**Kill all SSH tunnels:**
```bash
pkill -f "ssh.*mind-access00"
```

**Alternative single-tunnel for Ollama only (from mentor docs):**
```bash
ssh -L 11434:gpu01.mind.cs.umd.edu:11434 <username>@mind-access00.cs.umd.edu
```
This binds the remote Ollama port to local 11434. Keep this terminal open (or use `-f -N` for background).

### 6.3 JupyterHub

| URL | Notes |
|-----|-------|
| `https://gpuyter.mind.cs.umd.edu` | New JupyterHub (preferred for AI/LLM work) |
| `https://jupyter.mind.cs.umd.edu` | Old JupyterHub (being migrated) |

- Same login as SSH (UMD username + password)
- Priority access for AI/LLM workloads
- Runs in the same network as GPU nodes — can reach Ollama/vLLM directly without tunnels

### 6.4 VS Code Server (in-notebook)

From the JupyterHub launcher, click the "VS Code" tile. This runs a VS Code Server in the same environment as the notebook.

- Install packages in the VS Code integrated terminal
- Same filesystem as JupyterHub (shared home via GlusterFS)
- Can access GPU nodes directly (no SSH tunnels needed)

### 6.5 Web Proxy (for webapps)

If you run a webserver (e.g. Flask, FastAPI) from within the JupyterHub/VS Code environment:

**From JupyterHub notebook terminal:**
```
https://gpuyter.mind.cs.umd.edu/user/<your_id>/proxy/<your_port>
```

**From VS Code Server:**
```
https://gpuyter.mind.cs.umd.edu/user/<your_id>/vscode/proxy/<your_port>
```

Example: if your UMD username is `<username>` and the webserver listens on port 5150:
```
https://gpuyter.mind.cs.umd.edu/user/<username>/proxy/5150
```

---

## 7. Operational Procedures

### 7.1 Starting vLLM (gpu02)

> **Target (mentor-aligned):** serve vLLM from Docker on host port **11534**, with
> weights bind-mounted from `/scratch/models`. Compose + cutover runbook:
> [`deploy/vllm/`](deploy/vllm/README.md). **gpu01 stays Ollama-only** (`:11434`).
> Until cutover completes, the bare-metal commands below (port **8000**) remain the
> live path.

One model at a time (both L40S are needed via TP=2). Weights live under
`/scratch/models/`. Use a model-specific `TRITON_CACHE_DIR` and log file.

**Qwen3-30B-A3B-Instruct-2507-FP8 (CURRENT, since 2026-07-07):**

```bash
ssh gpu02.mind.cs.umd.edu

nohup env \
  TRITON_CACHE_DIR=/scratch/triton_cache_qwen3 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  CUDA_VISIBLE_DEVICES=0,1 \
  ~/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /scratch/models/qwen3-30b-a3b-instruct-2507-fp8 \
    --served-model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --tensor-parallel-size 2 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
  > /scratch/vllm_qwen3.log 2>&1 &
```

**Nemotron-30B FP8 (previous production model — exact restore command):**

```bash
nohup env \
  TRITON_CACHE_DIR=/scratch/triton_cache_nemotron \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  CUDA_VISIBLE_DEVICES=0,1 \
  ~/miniconda3/bin/python -m vllm.entrypoints.openai.api_server \
    --model /scratch/models/nemotron-nano-30b-fp8 \
    --served-model-name nvidia/nemotron-3-nano \
    --tensor-parallel-size 2 \
    --port 8000 \
    --host 0.0.0.0 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.85 \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --reasoning-parser-plugin /scratch/models/nemotron-nano-30b-fp8/nano_v3_reasoning_parser.py \
    --reasoning-parser nano_v3 \
  > /scratch/vllm_nemotron.log 2>&1 &
```

Remember when swapping: update `LLM_DEFAULT_MODEL` in the repo `.env` to the served
model name; Nemotron needs `NEMOTRON_THINKING=on` in run env (inert for other models).

**Key flags explained:**

| Flag | Why |
|------|-----|
| `TRITON_CACHE_DIR=/scratch/triton_cache_<model>` | Avoids GlusterFS Triton JIT race conditions (uses gpu02's 5.8 TB RAID0 `/scratch`) |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | Disables FlashInfer sampler (avoids compatibility issues) |
| `CUDA_VISIBLE_DEVICES=0,1` | Uses both L40S GPUs |
| `--tensor-parallel-size 2` | Splits model across 2 GPUs |
| `--max-model-len 131072` | 128K token context window |
| `--gpu-memory-utilization 0.85` | Leaves 15% VRAM headroom |

**Check startup:**
```bash
tail -f /scratch/vllm_qwen3.log
# Wait for "Application startup complete" message
```

**Stop vLLM:**
```bash
pkill -f "vllm.entrypoints"
```

### 7.2 Starting / Managing Ollama Models (gpu01)

Ollama runs as a persistent service on gpu01. You typically don't need to start/stop the daemon itself.

**List installed models:**
```bash
ssh gpu01.mind.cs.umd.edu 'ollama list'
```

**Pull a new model:**
```bash
ssh gpu01.mind.cs.umd.edu 'ollama pull <model_name>'
```

**Check Ollama daemon status:**
```bash
ssh gpu01.mind.cs.umd.edu 'systemctl status ollama 2>/dev/null || ps aux | grep ollama'
```

**Restart Ollama daemon (if needed):**
```bash
ssh gpu01.mind.cs.umd.edu 'sudo systemctl restart ollama'
```

### 7.3 SSH Tunnel Setup & Diagnostics

**Full tunnel setup script (run from your laptop):**
```bash
# Kill any existing tunnels
pkill -f "ssh.*mind-access00"

# Start vLLM tunnel
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 8000:localhost:8000 <username>@gpu02.mind.cs.umd.edu

# Start Ollama tunnel
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 11435:localhost:11434 <username>@gpu01.mind.cs.umd.edu

# Verify
sleep 2
lsof -iTCP:8000 -iTCP:11435 -sTCP:LISTEN
```

**Test tunnel connectivity:**
```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:11435/api/tags | python3 -m json.tool
```

**If tunnels drop:**
```bash
pkill -f "ssh.*mind-access00"
# Then re-run the tunnel setup commands above
```

### 7.4 Running the Pipeline

**Install the package first (required for module imports):**
```bash
pip install -e .
```

**Run with local GPU cluster:**
```bash
# Ensure .env is configured for local cluster (see Section 9)
# Ensure SSH tunnels are up (see Section 7.3)

python scripts/run_pipeline.py --input <your_doc.txt> --output output.json
```

---

## 8. Hard Rules & Gotchas

### Critical Rules

1. **NEVER run Ollama + vLLM on the same GPU node.**
   - vLLM with 32K KV-cache on 2x L40S consumes ~85% of VRAM.
   - Ollama loading an embedding model on the same node will cause OOM.
   - Ollama stays on gpu01, vLLM stays on gpu02. No exceptions.

2. **ALWAYS use the full HuggingFace / served-model ID for vLLM.**
   - Correct: `Qwen/Qwen3-30B-A3B-Instruct-2507` (match `/v1/models`)
   - Wrong: `qwen3:32b` (that's an Ollama tag, vLLM won't find it)

3. **ALWAYS set `TRITON_CACHE_DIR` to a local path.**
   - GlusterFS (`/home`) causes Triton JIT race conditions.
   - On gpu02: use `/scratch/triton_cache_<model>` (5.8 TB RAID0).
   - On gpu01: use `/tmp/triton_cache_<model>` (root NVMe, 4x 1.5 TB NVMe not mounted).

4. **Use port 11435 for Ollama tunnel, not 11434.**
   - Local port 11434 may be occupied by a local Ollama install.
   - The `.env` file expects `EMBEDDING_BASE_URL=http://localhost:11435/v1`.

5. **gpu02 root disk is nearly full (89%, only 8 GB free).**
   - Do NOT write large files to `/` or `/tmp` on gpu02.
   - Use `/scratch` (5.8 TB RAID0) for model weights, logs, and TRITON_CACHE_DIR.
   - If root fills up, vLLM and conda may break.

6. **gpu02 does NOT have `/fs/scratch` (NFS) mounted.**
   - Only gpu01 has the NFS shared scratch at `/fs/scratch`.
   - gpu02 has its own local `/scratch` (RAID0, 5.8 TB) which is faster but not shared.

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` when running scripts | Package not installed | `pip install -e .` from project root |
| vLLM OOM on startup | Another process using GPU memory | Check `nvidia-smi`, kill stale processes |
| Triton JIT hang/crash | GlusterFS cache conflict | Set `TRITON_CACHE_DIR=/tmp/...` |
| `Connection refused` on localhost:8000 | SSH tunnel dropped | Re-run tunnel setup (Section 7.3) |
| `Connection refused` on localhost:11435 | SSH tunnel dropped | Re-run tunnel setup (Section 7.3) |
| Empty LLM responses (Ollama) | Thinking model + JSON mode conflict | Code already handles this — see `openai_client.py` `_model_is_thinking()` |

---

## 9. Project Integration

### 9.1 `.env` Configuration

The project supports three LLM backends via `LLM_BACKEND` env var. For the local GPU cluster:

```bash
# ── Local GPU cluster config ──────────────────────────────
LLM_BACKEND=vllm
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY
LLM_DEFAULT_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507

EMBEDDING_BASE_URL=http://localhost:11435/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=mxbai-embed-large:335m
```

> **Current state:** The `.env` file currently points to Fireworks.ai (cloud). The local GPU cluster config is commented out. Uncomment it and comment out the Fireworks section when switching to local GPUs.

### 9.2 Backend Selection

The backend is selected in `multi_agent_kg/llm/openai_client.py`:

| `LLM_BACKEND` | Chat Client | Embedding Client |
|----------------|-------------|------------------|
| `ollama` | `OLLAMA_BASE_URL` (default `localhost:11434/v1`) | Same as chat (unless `EMBEDDING_BASE_URL` set) |
| `vllm` | `VLLM_BASE_URL` (e.g. `localhost:8000/v1`) | Separate `EMBEDDING_BASE_URL` if set, else same |
| `openai` | OpenAI API | OpenAI API (unless `EMBEDDING_BASE_URL` set) |

**Key design:** Chat and embeddings can target different servers. The typical local setup routes chat → vLLM (gpu02) and embeddings → Ollama (gpu01) via separate `EMBEDDING_BASE_URL`.

### 9.3 Model Specifications

Defined in `multi_agent_kg/core/adaptive_config.py` (`MODEL_SPECS` dict). Used for adaptive batch sizing.

| Model | Context | Max Output | Backend |
|-------|---------|------------|---------|
| `gemma4:31b` | 32768 | 8192 | Ollama |
| `google/gemma-4-31B-it` | 32768 | 8192 | vLLM |
| `Qwen/Qwen3-30B-A3B-Instruct-2507` | 131072 | — | vLLM (current on gpu02) |
| `qwen3:4b` / `qwen3:8b` | 32768 | 8192 | Ollama |
| `deepseek-r1:14b` | 32768 | 8192 | Ollama |
| `meta-llama/Llama-3.1-8B-Instruct` | 131072 | 32768 | vLLM |
| `meta-llama/Llama-3.3-70B-Instruct` | 131072 | 32768 | vLLM |

### 9.4 Thinking Model Handling

The codebase detects "thinking" models (gemma4, deepseek-r1, qwen3, etc.) that emit `<think>...</think>` reasoning blocks before JSON output. Key behaviors in `openai_client.py`:

- **Ollama:** JSON mode (`response_format: json_object`) is **disabled** for thinking models (GBNF grammar blocks `<think>` tokens → empty output)
- **vLLM:** JSON mode is tried on first attempt, falls back to unconstrained decoding if empty
- **Max tokens inflation:** `max_tokens * 3` for thinking models to leave room for visible output after reasoning
- **Robust JSON extraction:** `_extract_json()` handles markdown fences, `<think>` blocks (closed and unclosed), truncation repair

### 9.5 Embedding Model Notes

`mxbai-embed-large` is **asymmetric** — retrieval queries must be prefixed:
```
Represent this sentence for searching relevant passages: <query>
```

The code handles this automatically in `embed_query()` (`openai_client.py:722-732`). Passages are embedded raw (no prefix).

### 9.6 API Server + Web Frontend

The FastAPI backend (`scripts/api_server.py`) serves both the JSON API **and the built
React frontend** (`frontend/app/dist/`) from a single port. On gpu02 use port **5150**
— the default port (8000) is occupied by vLLM.

> **No node/npm on gpu02** — use `bun` (installed at `~/.bun/bin/bun`) for the frontend build.

**On gpu02 — build and start:**

```bash
conda activate agm
pip install -e ".[server]"                    # one-time: fastapi, uvicorn, python-multipart, pypdf

cd frontend/app
bun install                                    # one-time (and after dependency changes)
VITE_API_BASE='' bun run build                 # rebuild after any frontend change
cd ../..

nohup python scripts/api_server.py --port 5150 > pipeline_logs/api_server.log 2>&1 &
# add --data-only to skip QA/LLM initialization (KG view + upload only, /qa returns 503)
```

`VITE_API_BASE=''` makes the app call the API on the same origin it was served from,
which is what the single-port deploy needs. (Without it the app defaults to `/api`,
which only works behind the Vite dev proxy.)

**On your laptop — tunnel and open:**

```bash
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 5150:localhost:5150 <username>@gpu02.mind.cs.umd.edu
open http://localhost:5150
```

The UI shows: the knowledge graph (Explore), QA chat with streamed progress (QA),
governance review (Govern), document upload (+ INGEST), and a global status bar with
backend health, KG size, and a live per-stage pipeline progress bar during ingestion.

**Frontend dev mode** (only when actively editing frontend code): run
`cd frontend/app && bunx vite --port 3000` on gpu02 with the API on 5150 (the dev
proxy in `vite.config.js` targets 5150), and tunnel port 3000 instead.

Alternative access without a tunnel — the JupyterHub web proxy:
`https://gpuyter.mind.cs.umd.edu/user/<username>/proxy/5150/` (only works while your
JupyterHub session runs on the same node as the server; asset paths may need a
relative-base build: `VITE_API_BASE='' bun run build -- --base=./`).

---

## 10. Quick Reference

### One-liner: Start everything from your laptop

```bash
# 1. Start SSH tunnels
pkill -f "ssh.*mind-access00" 2>/dev/null
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 8000:localhost:8000 <username>@gpu02.mind.cs.umd.edu
ssh -f -N -J <username>@mind-access00.cs.umd.edu -L 11435:localhost:11434 <username>@gpu01.mind.cs.umd.edu

# 2. Verify
curl -s localhost:8000/v1/models | python3 -m json.tool
curl -s localhost:11435/api/tags | python3 -m json.tool
```

### One-liner: Check GPU status

```bash
ssh gpu01.mind.cs.umd.edu 'nvidia-smi'
ssh gpu02.mind.cs.umd.edu 'nvidia-smi'
```

### One-liner: Check running services

```bash
ssh gpu01.mind.cs.umd.edu 'ps aux | grep -E "ollama" | grep -v grep; curl -s localhost:11434/api/tags'
ssh gpu02.mind.cs.umd.edu 'ps aux | grep -E "vllm" | grep -v grep; curl -s localhost:8000/v1/models'
```

### One-liner: Switch .env to local GPU cluster

```bash
# Comment out Fireworks, uncomment local cluster
sed -i.bak \
  -e 's/^LLM_BACKEND=vllm$/# &/' \
  -e 's/^VLLM_BASE_URL=https:\/\/api.fireworks.ai.*/# &/' \
  -e 's/^VLLM_API_KEY=fw_.*/# &/' \
  -e 's/^LLM_DEFAULT_MODEL=accounts.*/# &/' \
  -e 's/^EMBEDDING_BASE_URL=https:\/\/api.fireworks.ai.*/# &/' \
  -e 's/^EMBEDDING_API_KEY=fw_.*/# &/' \
  -e 's/^EMBEDDING_MODEL=accounts.*/# &/' \
  -e 's/^# LLM_BACKEND=vllm$/LLM_BACKEND=vllm/' \
  -e 's/^# VLLM_BASE_URL=http:\/\/localhost:8000/VLLM_BASE_URL=http:\/\/localhost:8000/' \
  -e 's/^# VLLM_API_KEY=EMPTY/VLLM_API_KEY=EMPTY/' \
  -e 's/^# LLM_DEFAULT_MODEL=Qwen\/Qwen3-30B-A3B-Instruct-2507/LLM_DEFAULT_MODEL=Qwen\/Qwen3-30B-A3B-Instruct-2507/' \
  -e 's/^# LLM_DEFAULT_MODEL=google\/gemma-4-31B-it/LLM_DEFAULT_MODEL=Qwen\/Qwen3-30B-A3B-Instruct-2507/' \
  -e 's/^# EMBEDDING_BASE_URL=http:\/\/localhost:11435/EMBEDDING_BASE_URL=http:\/\/localhost:11435/' \
  -e 's/^# EMBEDDING_API_KEY=ollama/EMBEDDING_API_KEY=ollama/' \
  -e 's/^# EMBEDDING_MODEL=mxbai-embed-large:335m/EMBEDDING_MODEL=mxbai-embed-large:335m/' \
  .env
```

### One-liner: Kill everything and clean up

```bash
# Kill SSH tunnels
pkill -f "ssh.*mind-access00"

# Kill vLLM on gpu02
ssh gpu02.mind.cs.umd.edu 'pkill -f "vllm.entrypoints"'

# Check GPU memory is freed
ssh gpu01.mind.cs.umd.edu 'nvidia-smi'
ssh gpu02.mind.cs.umd.edu 'nvidia-smi'
```

### Port Reference

| Port | Service | Node | Local Tunnel Port |
|------|---------|------|-------------------|
| 11434 | Ollama | gpu01 | 11435 (via tunnel) |
| 8000 | vLLM | gpu02 | 8000 (via tunnel) |
| 5150 | FastAPI (API + web frontend) | gpu02 | 5150 (via tunnel) — see §9.6 |

### Login Reference

| Service | URL | Username |
|---------|-----|----------|
| SSH (jump) | `<username>@mind-access00.cs.umd.edu` | your UMD username |
| JupyterHub (new) | `https://gpuyter.mind.cs.umd.edu` | your UMD username |
| JupyterHub (old) | `https://jupyter.mind.cs.umd.edu` | your UMD username |
| VS Code Server | via JupyterHub launcher | your UMD username |

> All services use the same UMD username and password.
