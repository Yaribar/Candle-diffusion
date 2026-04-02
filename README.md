### Candle Stable Diffusion

Generate images from text prompts using Stable Diffusion, powered by
[Candle](https://github.com/huggingface/candle) — Hugging Face's pure-Rust ML
framework.

#### Why Candle instead of tch/libtorch?

| | **tch (libtorch)** | **Candle** |
|---|---|---|
| C++ dependency | ~2-3 GB libtorch download | None — pure Rust |
| Apple Silicon GPU | Not supported | Native Metal via `--features metal` |
| Build time | Long (C++ linking) | Normal Rust build |
| Weight download | Manual scripts | Automatic via `hf-hub` on first run |

#### Quick Start (Native)

```bash
# CPU (any platform)
cargo run --release -- --prompt "A rusty robot on a beach"

# Apple Silicon with Metal GPU
cargo run --release --features metal -- --prompt "A rusty robot on a beach"
```

Or use the Makefile:

```bash
make run                   # CPU
make run-metal             # Metal GPU (Mac)
```

#### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--prompt` | "A very realistic photo..." | Text prompt |
| `--sd-version` | `v1-5` | Model: `v1-5`, `xl`, `turbo` |
| `--n-steps` | 30 (1 for turbo) | Diffusion steps |
| `--cpu` | off | Force CPU |
| `--use-f16` | off | Half-precision weights |
| `--seed` | random | Reproducible generation |
| `--guidance-scale` | 7.5 (0 for turbo) | Classifier-free guidance |
| `--final-image` | `sd_final.png` | Output file path |
| `--height` / `--width` | varies by model | Image dimensions (divisible by 8) |
| `--num-samples` | 1 | Number of images to generate |

#### Docker (CPU only)

Docker Desktop on macOS cannot pass through Metal GPU, so the container runs on CPU.

```bash
# Build
docker build -t candle-diffusion .

# Run (mount a volume so model weights persist across runs)
docker run --rm -v hf-cache:/root/.cache/huggingface \
  candle-diffusion --prompt "A rusty robot on a beach"
```

#### GPU Availability

| Environment | GPU Backend | How |
|---|---|---|
| Mac native | Metal | `cargo build --features metal` |
| Linux native (NVIDIA) | CUDA | `cargo build --features cuda` |
| Docker on Mac | None (CPU) | Metal not available in containers |
| GitHub Codespace (default) | None (CPU) | No GPU attached |
| GPU Codespace (NVIDIA) | CUDA | `cargo build --features cuda` |

#### Disk Space

No libtorch download needed. Model weights (~5 GB for SD v1.5) are
automatically downloaded on first run to `~/.cache/huggingface/` via `hf-hub`.
