# Heuristic Bot Research

## Methodology

- Baseline opponent: `baseline` (`heuristic14`, naive play-first bot)
- Match configuration: 100000 games, stock size 20, max turns 10000
- Command template: `cargo run --quiet --release --bin winrate -- --games 100000 --no-chart --max-turns 10000 --stock-size 20 <bot> heuristic14`

## Results Overview

| Alias                    | Internal ID | Win% vs baseline | baseline Win% | Draw% | Notes                                                           |
| ------------------------ | ----------- | ---------------- | ------------- | ----- | --------------------------------------------------------------- |
| combo architect          | heuristic13 | 95.71%           | 4.29%         | 0.00% | Deep discard-assisted hand clears achieve best score.           |
| hand-cycle tactician     | heuristic11 | 95.23%           | 4.73%         | 0.04% | Full-hand sequence finder adds marginal gains.                  |
| duplication guardian     | heuristic10 | 94.98%           | 4.92%         | 0.10% | Preserves duplicated next-values without losing edge.           |
| situational blocker      | heuristic9  | 94.98%           | 4.92%         | 0.10% | Only blocks in fallback; regains strong win rate.               |
| high-value opportunist   | heuristic4  | 95.14%           | 4.78%         | 0.08% | Adds aggressive high-value plays once stock path blocked.       |
| late-game hoarder        | heuristic5  | 94.98%           | 4.92%         | 0.10% | Restricts to ≥6 plays; nearly ties the opportunist.             |
| mid-value opener         | heuristic6  | 95.06%           | 4.85%         | 0.09% | Threshold lowered to ≥5 without hurting results.                |
| top-heavy hoarder        | heuristic7  | 94.90%           | 5.00%         | 0.11% | Threshold raised to ≥7; slight regression vs. mid-value opener. |
| discard-agnostic planner | heuristic3  | 92.08%           | 7.58%         | 0.33% | Priority-free discards make planning more effective.            |
| stock-path planner       | heuristic2  | 85.53%           | 12.89%        | 1.58% | Stock-first planning lifts results despite minor draw rate.     |
| prioritized play planner | heuristic1  | 78.42%           | 21.58%        | 0.00% | Weighted play-score heuristic beats baseline comfortably.       |
| defensive blocker        | heuristic8  | 20.88%           | 73.03%        | 6.09% | Defensive blocks cripple offense; baseline dominates.           |
| prefer hand over discard | heuristic15 | 47.50%           | 52.50%        | 0.00% | Stock > hand > discard priority still loses to baseline.        |
| discard manager          | heuristic16 | 77.36%           | 22.63%        | 0.00% | Baseline play order with tuned discard scoring.                 |
| build-pile chooser       | heuristic17 | 45.52%           | 54.48%        | 0.00% | Adds pile-target ranking to the prefer-hand baseline.           |
| discard-aware planner    | heuristic18 | 77.48%           | 22.52%        | 0.00% | Stock chooser anticipates discard follow-ups.                   |
| baseline                 | heuristic14 | —                | —             | —     | Reference bot used for comparisons.                             |

## Prioritized Play Planner (heuristic1)

- Strategy notes: Weighted action scoring prefers stock plays, then discard, then hand while rewarding pile progress and tidy discard stacks.
- Baseline comparison: 78.42% win rate vs 21.58% for `baseline` (draws 0.00%).

## Stock-Path Planner (heuristic2)

- Strategy notes: Plans a minimal sequence to free the stock each turn; falls back to Heuristic 1 style discard scoring when stock progress is impossible.
- Baseline comparison: 85.53% win rate vs 12.89% for `baseline` (draws 1.58%).

## Discard-Agnostic Planner (heuristic3)

- Strategy notes: Same stock-first planner as Heuristic 2 but removes card-priority bias in discard scoring to prioritize shallow, duplicate-friendly discard piles.
- Baseline comparison: 92.08% win rate vs 7.58% for `baseline` (draws 0.33%).

## High-Value Opportunist (heuristic4)

- Strategy notes: After the stock plan, aggressively plays any card valued at least as high as the stock card to advance build piles before discarding.
- Baseline comparison: 95.14% win rate vs 4.78% for `baseline` (draws 0.08%).

## Late-Game Hoarder (heuristic5)

- Strategy notes: Mirrors Heuristic 4 but limits fallback plays to numbers ≥6 and strictly above the stock value to keep key cards in reserve.
- Baseline comparison: 94.98% win rate vs 4.92% for `baseline` (draws 0.10%).

## Mid-Value Opener (heuristic6)

- Strategy notes: Adjusts the aggressive-play threshold to numbers ≥5 while keeping the stock-first plan and discard logic from Heuristic 5.
- Baseline comparison: 95.06% win rate vs 4.85% for `baseline` (draws 0.09%).

## Top-Heavy Hoarder (heuristic7)

- Strategy notes: Tightens the fallback play rule to numbers ≥7, otherwise mirrors Heuristic 5/6 behavior.
- Baseline comparison: 94.90% win rate vs 5.00% for `baseline` (draws 0.11%).

## Defensive Blocker (heuristic8)

- Strategy notes: Builds on Heuristic 5 but vetoes plays that would make the next player's stock immediately playable, trading offense for defense.
- Baseline comparison: 20.88% win rate vs 73.03% for `baseline` (draws 6.09%).

## Situational Blocker (heuristic9)

- Strategy notes: Matches Heuristic 8 but only applies the opponent-protection veto as a last resort, restoring aggressive follow-up plays.
- Baseline comparison: 94.98% win rate vs 4.92% for `baseline` (draws 0.10%).

## Duplication Guardian (heuristic10)

- Strategy notes: Extends Heuristic 9 by refusing plays that break duplicate next-values across build piles, aiming to limit the opponent's options.
- Baseline comparison: 94.98% win rate vs 4.92% for `baseline` (draws 0.10%).

## Hand-Cycle Tactician (heuristic11)

- Strategy notes: Adds a search that plays the entire hand in sequence (without discards) when possible, triggering an immediate refill while preserving duplication rules.
- Baseline comparison: 95.23% win rate vs 4.73% for `baseline` (draws 0.04%).

## Staged Discard Sculptor (heuristic12)

- Strategy notes: Retains Heuristic 11 logic but tweaks discard scoring to reward placing n beneath n+1 on discard piles for future sequencing.
- Baseline comparison: 94.51% win rate vs 5.46% for `baseline` (draws 0.03%).

## Combo Architect (heuristic13)

- Strategy notes: Generalizes the full-hand planner to also consume discard piles during the sequence, enabling deeper combo turns while honoring duplication safeguards.
- Baseline comparison: 95.71% win rate vs 4.29% for `baseline` (draws 0.00%).

## Baseline (heuristic14)

- Strategy notes: Naive play-first policy—prioritize any play (stock, discard, then hand), otherwise end turn, and discard only when forced.
- Baseline comparison: —

## Prefer Hand Over Discard (heuristic15)

- Strategy notes: Simplistic priority ordering (stock > hand > discard) on top of Heuristic 14's always-play stance.
- Baseline comparison: 47.50% win rate vs 52.50% for `baseline` (draws 0.00%).

## Discard Manager (heuristic16)

- Strategy notes: Retains the baseline play-first policy but scores discard locations to keep stacks shallow and duplicate-friendly.
- Baseline comparison: 77.36% win rate vs 22.63% for `baseline` (draws 0.00%).

## Build-Pile Chooser (heuristic17)

- Strategy notes: Starts from the prefer-hand ordering, breaking ties between legal build targets to push deeper or nearly complete piles.
- Baseline comparison: 45.52% win rate vs 54.48% for `baseline` (draws 0.00%).

## Discard-Aware Planner (heuristic18)

- Strategy notes: Uses the discard manager's tidy piles while steering stock plays toward build piles that unlock immediate follow-ups from hand or discard.
- Baseline comparison: 77.48% win rate vs 22.52% for `baseline` (draws 0.00%).

