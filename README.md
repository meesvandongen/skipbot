# SkipBot

This project trains simple Skip-Bo policy networks using the Burn framework.

## Training

- Default training:

```powershell
make train
```

This will:

- Collect self-play data and train a small policy network.
- Save checkpoints under `checkpoints/` (e.g. `policy-bot-01.bin`).
- Write Burn dashboard logs per epoch under `checkpoints/burn-run-bot-XX/{train,valid}/epoch-N/`.

Current make defaults (tuned for a reasonably sized run without exhausting memory):

- `--games 64` per bot
- `--epochs 10`
- `--bots 1`
- `--batch-size 32`
- `--players 4`
- `--validation-split 0.1`
- `--exploration 0.05`
- `--max-turns 2000` (safety cap to prevent pathological long games)

If you want a larger run, override the defaults (be mindful of memory):

```powershell
$env:TRAIN_ARGS = "--games 64 --epochs 10 --bots 1 --batch-size 32"; make train
```

## Burn Dashboard

The training writes metric logs in the format used by Burn's built-in dashboard:

- Train logs: `checkpoints/burn-run-bot-XX/train/epoch-N/Loss.log`
- Valid logs: `checkpoints/burn-run-bot-XX/valid/epoch-N/Loss.log`

You can point Burn's dashboard viewer at the `checkpoints/burn-run-bot-XX` directory to visualize the learning curves. Refer to Burn's documentation for launching the dashboard viewer.

## Win-rate benchmarking and charts

You can simulate many games between any mix of bots and render a bar chart of per-seat win rates:

```powershell
cargo run --release --bin winrate -- --games 200 heuristic random policy --out winrates.png --max-turns 2000
```

Notes:

- Bots are specified as positional args; supported: `heuristic`, `random[:seed]`, `policy[:hidden[xdepth]]`.
- The chart is written to `--out` (PNG). A textual summary is always printed.
- Long games: Untrained `policy` bots can lead to very long games. Use `--max-turns` to cap per-game turns; any game that hits the cap is treated as a draw and excluded from win counts (reported separately).
- For human players, use the `simulate` binary instead (the `winrate` binary does not support interactive players).

Quick sanity checks:

```powershell
# Heuristic vs Random (fast)
cargo run --bin winrate -- --games 50 heuristic random

# Include an untrained Policy bot and cap turns to avoid stalls
cargo run --bin winrate -- --games 20 heuristic random policy --max-turns 1000
```

## Troubleshooting

- Memory: On Windows, large datasets can exhaust memory. Use smaller `--games`, `--bots`, and `--batch-size` values (see Makefile defaults). Start small, then scale up gradually. The default `--max-turns 2000` cap helps avoid runaway games; you can increase it if you need longer trajectories.
- Reproducibility: Control randomness with `--seed`.
- Win-rate runs: If you see "aborted" games in the win-rate summary, increase `--max-turns` or omit untrained `policy` bots.
- Output directory: Use `--output <dir>` to change where checkpoints and dashboard logs are written.
