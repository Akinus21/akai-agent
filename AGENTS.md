# AGENTS.md — akai-agent

## Identity
Repo: `Akinus21/akai-agent`
Binary: `akai-agent`
Purpose: Cross-platform Rust binary that turns any GPU machine into a remote
         worker for the akai-net distributed inference system.

## SSH Key
All git operations use: `/config/.ssh/github`

Always push like this:
```bash
GIT_SSH_COMMAND="ssh -i /config/.ssh/github" git push origin main
```

Set for the session:
```bash
export GIT_SSH_COMMAND="ssh -i /config/.ssh/github"
```

## Always Push When Done
After every meaningful change, push to main. CI runs automatically.
Build results arrive via webhook.

## Project Location
`/home/opencode/projects/akai-agent`

## Build
```bash
cargo build --release
# Binary: target/release/akai-agent
```

## Docker Socket
The opencode container has access to `/var/run/docker.sock`.
You can run docker commands if needed for testing or inspecting the stack:
```bash
docker ps
docker logs akai-net
docker exec akai-net switch-model -f /models/foo.gguf
```

## New Architecture (v2)

The new architecture uses direct TCP connection to the akai-net hub:

```
Worker (this binary)
  ↓ connects to hub:50051 via TCP
  ↓ sends: HubMessage::Register
  ← receives: HubMessage::InferenceRequest
  ↓ processes layers, returns hidden states
Akai-Net Hub
  ← OpenAI-compatible API on :8080
  ← coordinates pipeline across workers
```

### New Worker Protocol

Workers connect via TCP to port 50051. Protocol is simple JSON messages:

| Message | Direction | Description |
|---------|-----------|-------------|
| `HubMessage::Register` | Worker→Hub | Worker announces capabilities |
| `HubMessage::Heartbeat` | Worker→Hub | Periodic alive check |
| `HubMessage::InferenceRequest` | Hub→Worker | Tokens to process |
| `HubMessage::InferenceResponse` | Worker→Hub | Token + hidden states |

### Start Worker Command

```bash
# New command (v2 protocol):
akai-agent start-worker \
  --hub-addr 192.168.1.100:50051 \
  --worker-id desktop-1 \
  --model-path /models/model.gguf \
  --layer-offset 0 \
  --num-layers 16
```

Arguments:
- `--hub-addr`: Hub address (IP:50051)
- `--worker-id`: Unique worker identifier
- `--model-path`: Path to GGUF model file
- `--layer-offset`: Starting layer for this worker
- `--num-layers`: Number of layers this worker handles

### Layer Assignment

Workers are sorted by capacity (GPU priority):
- GPU workers: score = vram_gb * 100
- CPU workers: score = 1

Weakest worker gets first layers, strongest gets last layers.

## Legacy Architecture (v1)

The old architecture used ollama-queue as a registry and WireGuard tunnels:

```
akai-agent (v1)
  ↓ init: provision WireGuard, download rpc-server, register
  ↓ start: heartbeat every 30s, respawn rpc-server if dead
ollama-queue (Python FastAPI on VPS)
  ↓ holds live worker registry
  ↓ GET /workers returns rpc_string e.g. "10.8.0.2:50052"
```

Commands: `init`, `start`, `install`, `status`, `update-rpc`, `petals-start`

## Source Files
| File | Purpose |
|---|---|
| `src/main.rs` | Entry point |
| `src/cli.rs` | All CLI commands + handlers |
| `src/config.rs` | Config struct, load/save, paths |
| `src/gpu.rs` | GPU detection (nvidia-smi / rocm-smi) |
| `src/rpc.rs` | Download, version-check, spawn rpc-server |
| `src/build.rs` | Build rpc-server from source (CUDA detection, distrobox support) |
| `src/queue_client.rs` | HTTP client for ollama-queue registry API |
| `src/wireguard/` | Per-OS WireGuard setup (linux/macos/windows) |
| `src/service/` | systemd / launchd / Windows Service installers |
| `src/worker.rs` | New v2 worker protocol implementation |

## Installation (end users)
```bash
# macOS / Linux
brew tap akinus21/homebrew-tap
brew install akai-agent

# Windows (native, no WSL needed)
# Download akai-agent-windows-x86_64.exe from:
# https://github.com/Akinus21/akai-agent/releases/latest

# Any platform from source
cargo install --git https://github.com/Akinus21/akai-agent
```

## Config File
Linux/macOS: ~/.config/akai-agent/config.toml
Windows:     %APPDATA%\akai-agent\config.toml

Fields (v1): queue_url, api_key, worker_id, worker_name, wg_ip, wg_peer_id,
            rpc_port, gpu, vram_gb, rpc_binary, rpc_version
