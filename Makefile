# SkipBot project makefile providing convenient cargo commands

CARGO ?= cargo
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

simulate:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin simulate -- $(SIM_ARGS)

winrate:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin winrate -- $(WINRATE_ARGS)

play:
	$(CARGO) run --release $(CARGO_FLAGS) $(FEATURE_FLAGS) --bin simulate -- human heuristic heuristic heuristic --visualize

bench:
	$(CARGO) bench $(CARGO_FLAGS) $(FEATURE_FLAGS)
