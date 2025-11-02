# SkipBot project makefile providing convenient cargo commands

CARGO ?= cargo
TRAIN_ARGS ?= --games 16 --epochs 5 --bots 1 --players 4 --batch-size 16 --validation-split 0.1 --exploration 0.05
SIM_ARGS ?= --games 100 --players 4
FEATURES ?=
TARGET ?=

CARGO_FLAGS := $(if $(TARGET),--target $(TARGET),)
FEATURE_FLAGS := $(if $(FEATURES),--features $(FEATURES),)

.PHONY: build release check test fmt fmt-check clippy doc clean train simulate bench

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

simulate:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin simulate -- $(SIM_ARGS)

bench:
	$(CARGO) bench $(CARGO_FLAGS) $(FEATURE_FLAGS)
