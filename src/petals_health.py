#!/usr/bin/env python3
"""
Petals Health Server - lightweight HTTP health endpoint alongside Petals.

Petals CLI runs the actual inference server. This provides a separate
health/monitoring endpoint for the hub to check worker status.
"""

import asyncio
import argparse
import logging
import subprocess
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("petals-health")


async def run_petals_server(model_name: str, port: int, quantize: str = None, max_length: int = 4096):
    """Start the Petals server as a subprocess."""
    cmd = [
        sys.executable, "-m", "petals.cli.run_server",
        model_name,
        "--port", str(port),
        "--max-length", str(max_length),
    ]
    if quantize:
        cmd.extend(["--quantize", quantize])

    log.info(f"Starting Petals: {' '.join(cmd)}")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )

    return process


async def health_monitor(process, port):
    """Monitor the Petals process and log output."""
    if process.stdout is None:
        return

    buffer = ""
    while True:
        try:
            char = await process.stdout.read(1)
            if not char:
                break

            char = char.decode('utf-8', errors='replace')
            buffer += char

            if '\n' in buffer:
                lines = buffer.split('\n')
                buffer = lines[-1]

                for line in lines[:-1]:
                    log.info(f"[petals] {line.strip()}")
        except Exception as e:
            log.error(f"Monitor error: {e}")
            break

    await process.wait()


async def main():
    parser = argparse.ArgumentParser(description="Petals Health Server")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--port", type=int, default=50052, help="Petals server port")
    parser.add_argument("--health-port", type=int, default=50053, help="Health endpoint port")
    parser.add_argument("--quantize", help="Quantization (e.g., int8)")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--public-name", default="", help="Public name for swarm")
    args = parser.parse_args()

    petals_process = await run_petals_server(
        args.model, args.port, args.quantize, args.max_length
    )

    monitor_task = asyncio.create_task(health_monitor(petals_process, args.port))

    from aiohttp import web

    async def health(request):
        status = "running" if petals_process.returncode is None else "stopped"
        return web.json_response({
            "status": status,
            "model": args.model,
            "port": args.port,
            "pid": petals_process.pid if petals_process else None,
        })

    async def ready(request):
        if petals_process and petals_process.returncode is None:
            return web.json_response({"ready": True})
        return web.json_response({"ready": False}, status=503)

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.health_port)
    await site.start()

    log.info(f"Health server on 0.0.0.0:{args.health_port}")
    log.info(f"Petals server on 0.0.0.0:{args.port}")

    try:
        await asyncio.Future()
    except asyncio.CancelledError:
        pass
    finally:
        monitor_task.cancel()
        petals_process.terminate()
        await petals_process.wait()


if __name__ == "__main__":
    asyncio.run(main())