# SkipBot project makefile providing convenient cargo commands

CARGO ?= cargo
# Approx ~5 minute default training configuration (collection + optimization) on a typical modern CPU.
# Adjust GAMES to scale dataset size; higher BATCH improves throughput but increases memory.
# Patience enables early stopping if validation stagnates.
TRAIN_ARGS ?= --games 450 --epochs 1000 --bots 1 --players 4 --batch-size 64 --validation-split 0.1 --exploration 0.05 --max-turns 1500 --stock-size 3 --patience 20
RESUME_CHECKPOINT ?= checkpoints/policy-bot-01-best.bin
# Resume continues training for additional epochs using same architecture & dataset seed.
RESUME_ARGS ?= --games 450 --epochs 1000 --batch-size 64 --patience 20
# Default simulate args: provide 4 non-interactive bots to avoid blocking for human input
# You can override by running: make simulate SIM_ARGS="heuristic random policy:128x3 heuristic --max-turns 2000"
SIM_ARGS ?= heuristic heuristic heuristic heuristic --max-turns 2000 --visualize
WINRATE_ARGS ?= --games 200 --max-turns 2000 heuristic heuristic heuristic heuristic
FEATURES ?=
TARGET ?=

CARGO_FLAGS := $(if $(TARGET),--target $(TARGET),)
FEATURE_FLAGS := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: build release check test fmt fmt-check clippy doc clean train train-resume simulate winrate play bench

build:
	$(CARGO) build $(CARGO_FLAGS) $(FEATURE_FLAGS)

release:
	$(CARGO) build --release $(CARGO_FLAGS) $(FEATURE_FLAGS)

check:
	$(CARGO) check $(CARGO_FLAGS) $(FEATURE_FLAGS)

test:
	$(CARGO) test $(CARGO_FLAGS) $(FEATURE_FLAGS)

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt -- --check

clippy:
	$(CARGO) clippy --all-targets $(FEATURE_FLAGS)

doc:
	$(CARGO) doc --all-features --no-deps

clean:
	$(CARGO) clean

train:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin train -- $(TRAIN_ARGS)

train-resume:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin train -- --resume $(RESUME_CHECKPOINT) $(RESUME_ARGS)

simulate:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin simulate -- $(SIM_ARGS)

winrate:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin winrate -- $(WINRATE_ARGS)

play:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin simulate -- human heuristic heuristic heuristic --visualize

bench:
	$(CARGO) bench $(CARGO_FLAGS) $(FEATURE_FLAGS)
