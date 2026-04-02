PROMPT ?= "A very realistic photo of a rusty robot walking on a sandy beach"

.PHONY: build build-metal run run-metal docker-build docker-run format lint test clean

build:
	cargo build --release

build-metal:
	cargo build --release --features metal

run:
	cargo run --release -- --prompt $(PROMPT)

run-metal:
	cargo run --release --features metal -- --prompt $(PROMPT)

docker-build:
	docker build -t candle-diffusion .

docker-run:
	docker run --rm -v hf-cache:/root/.cache/huggingface candle-diffusion --prompt $(PROMPT)

format:
	cargo fmt

lint:
	cargo clippy -- -D warnings

test:
	cargo test

clean:
	cargo clean
