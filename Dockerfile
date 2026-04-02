## Builder stage
FROM rust:1.85-slim AS builder

RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml ./
COPY src/ ./src/

RUN cargo build --release

## Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/candle-diffusion /usr/local/bin/candle-diffusion

# Model weights are cached here — mount a volume for persistence:
#   docker run -v hf-cache:/root/.cache/huggingface ...
ENV HF_HOME=/root/.cache/huggingface

ENTRYPOINT ["candle-diffusion"]
CMD ["--prompt", "A very realistic photo of a rusty robot walking on a sandy beach"]
