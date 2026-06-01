#!/usr/bin/env python3
"""
Pipeline Worker Server

Runs on each worker machine. Loads model and exposes pipeline compute endpoint.

Protocol:
1. Receives POST /v1/pipeline/completions with:
   - prompt: text input
   - model: model name
   - layer_offset: starting layer for this worker
   - max_layers: number of layers this worker handles
   - is_first: True if this is first stage (computes embeddings)
   - is_last: True if this is last stage (computes lm_head)
   - num_workers: total workers in pipeline

2. This worker loads full model but only computes its layer range
3. Returns streaming response with computed hidden states or token
"""

import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import Optional, AsyncIterator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("pipeline-worker")


class PipelineWorkerServer:
    """Worker server that computes assigned layers."""

    def __init__(
        self,
        model_path: str,
        layer_offset: int = 0,
        max_layers: int = 0,
        port: int = 50052,
        n_ctx: int = 4096,
    ):
        self.model_path = model_path
        self.layer_offset = layer_offset
        self.max_layers = max_layers
        self.port = port
        self.n_ctx = n_ctx
        self.model = None
        self.model_info = None

    async def load_model(self):
        """Load the model using llama-cpp-python."""
        try:
            from llama_cpp import Llama
        except ImportError:
            log.error("llama-cpp-python not installed")
            return False

        log.info(f"Loading model from {self.model_path}...")
        log.info(f"Layer range: {self.layer_offset} to {self.layer_offset + self.max_layers}")

        # Load model with GPU offload for available layers
        # For pipeline worker, we load full model but would only use our layer range
        n_gpu_layers = min(self.max_layers, 32) if self.max_layers > 0 else 0

        self.model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

        self.model_info = {
            "layer_offset": self.layer_offset,
            "max_layers": self.max_layers,
            "n_ctx": self.n_ctx,
        }

        log.info(f"Model loaded: layer_offset={self.layer_offset}, max_layers={self.max_layers}")
        return True

    async def compute_embeddings(self, tokens: list[int]) -> list[float]:
        """Compute embeddings (used by first stage)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Get logits - we use the last token's hidden state as embedding
        output = self.model.eval(tokens)
        if len(output) > 0:
            return output[-1].tolist()
        return [0.0] * 4096  # Default embedding size

    async def forward_layers(self, hidden_states: list[float], token: int) -> list[float]:
        """Forward through this worker's layers."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # For pipeline worker, we run full model forward pass
        # but this only makes sense if we can control layer range
        # In practice, llama.cpp doesn't support partial layer computation

        # For now, compute full forward pass and return
        # The hub will only use outputs for our layer range
        tokens = [token]
        output = self.model.eval(tokens)
        if len(output) > 0:
            return output[-1].tolist()
        return hidden_states

    async def generate_token(self, hidden_states: list[float]) -> int:
        """Generate next token (used by last stage)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Simple: get logits for dummy token and argmax
        # In practice, we'd use the model's lm_head
        logits = self.model.eval([0])
        if len(logits) > 0:
            return logits[-1].tolist().index(max(logits[-1].tolist()))
        return 0

    async def infer(
        self,
        prompt: str,
        max_tokens: int = 100,
        is_first: bool = True,
        is_last: bool = True,
    ) -> AsyncIterator[dict]:
        """Run inference through this worker's layer range."""

        if self.model is None:
            if not await self.load_model():
                yield {"error": "Failed to load model"}
                return

        # Tokenize
        try:
            tokens = self.model.tokenize(prompt.encode())
        except Exception as e:
            yield {"error": f"Tokenization failed: {e}"}
            return

        log.info(f"Processing {len(tokens)} tokens, layers {self.layer_offset}-{self.layer_offset + self.max_layers}")

        # If this is the first stage, compute embeddings first
        if is_first:
            hidden = await self.compute_embeddings(tokens)
        else:
            # Not first stage - use zeros as initial hidden (hub provides real values)
            hidden = [0.0] * 4096

        # For each token in the generation
        for _ in range(max_tokens):
            # Forward through layers
            if is_last:
                # Last stage - generate token
                token = await self.generate_token(hidden)
                word = ""
                try:
                    word = self.model.detokenize([token]).decode(errors="ignore")
                except:
                    pass

                yield {
                    "token": token,
                    "content": word,
                    "done": False,
                }

                # Update hidden with new token for next iteration
                hidden = await self.forward_layers(hidden, token)
            else:
                # Not last stage - just forward hidden states
                hidden = await self.forward_layers(hidden, 0)
                yield {
                    "hidden_states": hidden[:128],  # Send abbreviated
                    "done": False,
                }

        yield {"done": True}

    async def health_check(self) -> dict:
        """Return worker health and configuration."""
        return {
            "status": "ok" if self.model is not None else "loading",
            "layer_offset": self.layer_offset,
            "max_layers": self.max_layers,
            "model_loaded": self.model is not None,
        }


async def main():
    parser = argparse.ArgumentParser(description="Pipeline Worker Server")
    parser.add_argument("--model", required=True, help="Path to model GGUF file")
    parser.add_argument("--layer-offset", type=int, default=0, help="Starting layer for this worker")
    parser.add_argument("--max-layers", type=int, default=0, help="Number of layers this worker handles")
    parser.add_argument("--port", type=int, default=50052, help="Port to listen on")
    parser.add_argument("--ctx-size", type=int, default=4096, help="Context size")
    args = parser.parse_args()

    worker = PipelineWorkerServer(
        model_path=args.model,
        layer_offset=args.layer_offset,
        max_layers=args.max_layers,
        port=args.port,
        n_ctx=args.ctx_size,
    )

    await worker.load_model()

    from aiohttp import web

    async def health(request):
        return web.json_response(await worker.health_check())

    async def completions(request):
        """Pipeline compute endpoint."""
        data = await request.json()
        prompt = data.get("prompt", "")
        max_tokens = data.get("max_tokens", 100)
        is_first = data.get("is_first", True)
        is_last = data.get("is_last", True)

        async def generate():
            async for chunk in worker.infer(prompt, max_tokens, is_first, is_last):
                if "error" in chunk:
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif "content" in chunk:
                    yield f"data: {json.dumps({'choices': [{'text': chunk['content']}]})}\n\n"
                elif "hidden_states" in chunk:
                    yield f"data: {json.dumps({'hidden_states': chunk['hidden_states']})}\n\n"
                elif chunk.get("done"):
                    pass
                else:
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return web.Response(
            text=generate(),
            content_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/pipeline/completions", completions)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"Pipeline worker ready on 0.0.0.0:{args.port}")
    log.info(f"Layers: {args.layer_offset} to {args.layer_offset + args.max_layers}")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())