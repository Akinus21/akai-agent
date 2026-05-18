# akai-agent

Remote GPU worker agent for the akai-net distributed inference system.

## Features

- Turns any GPU machine into a remote RPC worker
- Provisions WireGuard tunnel automatically
- Downloads and manages `rpc-server` binary from llama.cpp releases
- Registers with ollama-queue registry
- Heartbeat keep-alive every 30 seconds
- Auto-restarts `rpc-server` if it crashes
- Graceful shutdown on Ctrl+C

## Installation

```bash
brew tap akinus21/homebrew-tap
brew install akai-agent
```

Or from source:
```bash
cargo install --git https://github.com/Akinus21/akai-agent
```

## Setup

```bash
akai-agent init --queue https://ollama.akinus21.com --key YOUR_KEY
sudo akai-agent install
akai-agent status
```

## Usage

```bash
akai-agent init    # Provision WireGuard, download rpc-server, register
akai-agent start  # Run the worker daemon
akai-agent status # Show worker status
akai-agent update-rpc # Update rpc-server binary
sudo akai-agent install # Install as system service
```

## How It Works

1. `init` provisions a WireGuard tunnel to the VPS, downloads the `rpc-server` binary, and registers with ollama-queue
2. `start` runs the heartbeat loop and manages the `rpc-server` child process
3. akai-net queries ollama-queue for live workers and launches `llama-server --rpc <wg_ip>:50052`

## Requirements

- Linux, macOS, or Windows
- NVIDIA GPU (nvidia-smi) or AMD GPU (rocm-smi) — optional, CPU-only workers supported
- WireGuard tools installed