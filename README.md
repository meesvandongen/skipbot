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

## Troubleshooting

- Memory: On Windows, large datasets can exhaust memory. Use smaller `--games`, `--bots`, and `--batch-size` values (see Makefile defaults). Start small, then scale up gradually. The default `--max-turns 2000` cap helps avoid runaway games; you can increase it if you need longer trajectories.
- Reproducibility: Control randomness with `--seed`.
- Output directory: Use `--output <dir>` to change where checkpoints and dashboard logs are written.
