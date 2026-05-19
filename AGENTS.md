# AGENTS.md — akai-agent

## Identity
Repo: `Akinus21/akai-agent`
Binary: `akai-agent`
Purpose: Cross-platform Rust binary that turns any GPU machine into a remote
         RPC worker for the akai-net distributed inference system.

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

## How the System Works
akai-agent (this binary)
↓  init: provision WireGuard, download rpc-server, register
↓  start: heartbeat every 30s, respawn rpc-server if dead
ollama-queue (Python FastAPI on VPS)
↓  holds live worker registry
↓  GET /workers returns rpc_string e.g. "10.8.0.2:50052"
akai-net (Docker container on VPS)
↓  queries ollama-queue on startup for live workers
↓  launches: llama-server --rpc 10.8.0.2:50052 --model /models/...
↓  exposes /v1/chat/completions on :8080
ollama-queue proxy
↓  routes "akai/" requests → akai-net:8080
↓  routes "cloud/" requests → Cloud Ollama:11434

## Key Design Rules
- NO Ollama dependency — workers run `rpc-server` (from llama.cpp), NOT Ollama
- NO GPU crates (nvml-wrapper, cuda-sys) — shell out to nvidia-smi/rocm-smi only (gpu.rs)
- WireGuard AllowedIPs = 10.8.0.0/24 ONLY — never 0.0.0.0/0
- `rpc-server` binary is downloaded from ggml-org/llama.cpp GitHub Releases at runtime
- Workers bind rpc-server to 0.0.0.0 but it is only reachable via WireGuard
- Port 50052 must NEVER be exposed to the public internet
- Graceful shutdown on Ctrl+C: kill rpc-server child → DELETE /workers/{id} → exit
- Re-register automatically if heartbeat returns 404

## Source Files
| File | Purpose |
|---|---|
| `src/main.rs` | Entry point |
| `src/cli.rs` | All CLI commands + handlers |
| `src/config.rs` | Config struct, load/save, paths |
| `src/gpu.rs` | GPU detection (nvidia-smi / rocm-smi) |
| `src/rpc.rs` | Download, version-check, spawn rpc-server |
| `src/build.rs` | Build rpc-server from source (CUDA detection) |
| `src/queue_client.rs` | HTTP client for ollama-queue registry API |
| `src/wireguard/` | Per-OS WireGuard setup (linux/macos/windows) |
| `src/service/` | systemd / launchd / Windows Service installers |

## ollama-queue API (what this binary calls)
POST /workers/provision       → get WireGuard peer config
POST /workers/register        → register worker (id, wg_ip, rpc_port, gpu, vram_gb)
POST /workers/{id}/heartbeat  → keep-alive every 30s
DELETE /workers/{id}          → deregister on clean shutdown
GET  /workers/{id}            → status check (used by akai-agent status)
All endpoints require: `X-Worker-Key: <api_key>` header.
Heartbeat returning 404 means the worker fell out of registry → re-register automatically.

## rpc-server Binary Management
Three-tier approach (in order of priority):
1. **Pre-built CUDA bundle** (Linux x86_64 only): Downloaded from this repo's GitHub Release
   `akai-agent-rpc-cuda-linux-x86_64.tar.gz` — contains rpc-server + libggml*.so with CUDA
2. **Source build** (fallback, Linux x86_64 with NVIDIA GPU): Builds llama.cpp locally with `-DGGML_CUDA=ON`
   - Auto-installs build tools and CUDA toolkit if missing
   - Detects atomic/immutable distros (Silverblue) and uses Homebrew/rpm-ostree accordingly
3. **CPU-only download** (fallback for all platforms): Downloads pre-built from llama.cpp GitHub Releases

- Homebrew formula `post_install` extracts the CUDA bundle into `~/.local/share/akai-agent/`
- GitHub APIs: `Akinus21/akai-agent` releases (CUDA bundle) + `ggml-org/llama.cpp` releases (CPU-only)
- Always include header: `User-Agent: akai-agent/<version>`
- Stored at: ~/.local/share/akai-agent/rpc-server (Linux/macOS)
             %LOCALAPPDATA%\akai-agent\rpc-server.exe (Windows)
- CUDA .so files extracted to: ~/.local/share/akai-agent/lib/ (Linux)
- LD_LIBRARY_PATH set to lib/ dir when spawning rpc-server on Linux
- Daily update check in start loop; manual: `akai-agent update-rpc`
- Version stored in config.toml as `rpc_version`

## Platform Asset Mapping
| Platform | GitHub Release Asset |
|---|---|
| Linux x86_64 | llama-*-bin-ubuntu-x64.zip |
| Linux aarch64 | llama-*-bin-ubuntu-arm64.zip |
| macOS arm64 | llama-*-bin-macos-arm64.zip |
| macOS x86_64 | llama-*-bin-macos-x64.zip |
| Windows x86_64 | llama-*-bin-win-cuda-cu12.2.0-x64.zip |

## CI/CD
- `build.yml`: push to main → build linux x86_64 + build rpc-server with CUDA → GitHub Release → update Homebrew tap → webhook
- `release.yml`: tag push (auto from build.yml) → cross-compile 5 targets + build rpc-server CUDA bundle → upload all to release → full multi-platform Homebrew formula with rpc-cuda resource
- Webhook endpoint: `https://webhook.akinus21.com/webhook/akai-agent-build`
- On failure: full build log sent in `errors` field of webhook payload
- On success: tag and image info sent

## Required GitHub Secrets
| Secret | Purpose |
|---|---|
| `GH_TOKEN` | PAT with `contents:write` scope for tagging + releases |
| `TAP_TOKEN` | PAT with write access to `Akinus21/homebrew-tap` |
| `WEBHOOK_HMAC_SECRET` | Shared secret for HMAC-signing webhook payloads |

## Cross-Compilation Targets
| Target | Runner |
|---|---|
| x86_64-unknown-linux-gnu | ubuntu-latest |
| aarch64-unknown-linux-gnu | ubuntu-latest + gcc-aarch64-linux-gnu |
| x86_64-apple-darwin | macos-latest |
| aarch64-apple-darwin | macos-latest |
| x86_64-pc-windows-msvc | windows-latest |

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

## First-Time Setup (end users)
```bash
akai-agent init --queue https://ollama.akinus21.com --key YOUR_KEY
sudo akai-agent install
akai-agent status
```

## Config File
Linux/macOS: ~/.config/akai-agent/config.toml
Windows:     %APPDATA%\akai-agent\config.toml

Fields: queue_url, api_key, worker_id, worker_name, wg_ip, wg_peer_id,
        rpc_port, gpu, vram_gb, rpc_binary, rpc_version