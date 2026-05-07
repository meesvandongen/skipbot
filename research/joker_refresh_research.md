# Joker-Spending Tactic Experiment

When the active player holds Skip-Bo (joker) cards and Heuristic 13 wants to
use them in this turn's stock-progression / hand-clearing combo, is it
better to **spend** the jokers now or **save** them for a later turn?

This experiment varies a single tunable: the largest hand-joker count `N`
at which the bot is allowed to spend any joker on the current turn. If the
hand currently holds *more* jokers than `N`, every joker-using Play action
is suppressed for that turn — those jokers are saved for a future turn
instead.

Per the framing:

- `N = 1`: spend jokers only when the hand has exactly **1** joker.
- `N = 2`: spend jokers only when the hand has **1 or 2** jokers.
- `N = 5`: spend jokers whenever — equivalent to plain Heuristic 13, since
  hand size is 5 and so `joker_count ≤ 5` always.

The game rules are vanilla Skip-Bo throughout. No engine changes —
this is purely a bot-policy experiment.

## Bot Policy: `jokertactic:N`

The bot wraps Heuristic 13 (the strongest existing heuristic, "combo
architect", 95.71% baseline win rate) and gates its joker-using moves on
the per-turn budget `N`.

For every decision:

1. Count Skip-Bo cards in the current hand.
2. If `joker_count == 0` or `joker_count <= N`, defer to plain Heuristic 13.
   The combo plays — `can_play_stock`'s prefix sequence and
   `can_play_all_hand`'s full hand-clear, both of which may consume one or
   more jokers — are **allowed**.
3. Otherwise (`joker_count > N`): filter every Play action that would
   consume a Skip-Bo (from hand, stock top, or any discard top) out of
   `legal_actions` and re-run Heuristic 13 against the filtered list.
   Heuristic 13 will fall through to a non-joker play, a discard, or end
   turn as appropriate. The held jokers stay in the hand for the next
   turn.

This gives a clean experimental knob: as `N` grows, the bot becomes more
willing to spend jokers in this turn's combos.

## Methodology

- Match-up: `jokertactic:N` (under test) vs. `heuristic13` (control), 1 v 1.
- 100,000 games per scenario, seating permuted each game.
- `--stock-size 20`, `--max-turns 10000` (matches `heuristic_research.md`).
- Seed: default (`0xC0FFEE...5EED`), so all scenarios use identical decks
  and seating; only the joker budget changes between rows.
- At 100,000 games per cell the standard error of the win-rate estimate is
  ≈ √(0.25 / 100000) ≈ 0.16 pp; differences smaller than ~0.5 pp are noise.
- Command template:
  `cargo run --quiet --release --bin winrate -- --games 100000 --no-chart --max-turns 10000 --stock-size 20 jokertactic:<N> heuristic13`

## Results

| Scenario               | jokertactic win % | heuristic13 win % | Δ (jt − h13) |   jt decisions |  h13 decisions | Notes                                                                                              |
| ---------------------- | ----------------: | ----------------: | -----------: | -------------: | -------------: | -------------------------------------------------------------------------------------------------- |
| `jokertactic:1` vs h13 |             42.77 |             57.00 |       −14.23 |     10,337,116 |     10,629,474 | Bot saves jokers whenever it has ≥2 in hand. Joker-using combos suppressed often → big regression. |
| `jokertactic:2` vs h13 |             49.22 |             50.60 |        −1.38 |     10,330,941 |     10,372,591 | Saves jokers only when ≥3 in hand. Rare suppression — small but measurable hit (≈9× SE).           |
| `jokertactic:3` vs h13 |             50.10 |             49.73 |        +0.37 |     10,326,767 |     10,331,554 | Saves only when ≥4 in hand. Suppression event is uncommon; result within noise of plain h13.        |
| `jokertactic:4` vs h13 |             50.11 |             49.72 |        +0.39 |     10,324,961 |     10,328,222 | Saves only when hand is all 5 jokers. Negligible — within noise of plain h13.                       |
| `jokertactic:5` vs h13 |             50.11 |             49.72 |        +0.39 |     10,324,972 |     10,328,169 | Equivalent to plain h13 (joker_count ≤ 5 always). Sanity check.                                     |

(*"jt decisions" / "h13 decisions" are total `select_action` calls across
the 100,000 games. Note that the suppression mechanism actually *reduces*
`jokertactic`'s own per-turn decision count — when the joker-using combo is
skipped, the bot stalls into a single discard rather than chaining several
plays — but games run longer overall, so h13's totals climb. At N=1 h13
gains +301k decisions over the baseline (10.33M → 10.63M); at N=2 +44k;
and N≥3 is essentially zero.*)

## Interpretation

The headline answer to "should we use jokers in this turn's combo, or
save them?" is **spend them.** Using them in the same combos that
Heuristic 13 picks is the right play; the more we restrict that, the
worse we do, and the curve is monotonic in the budget `N`.

Concretely:

1. **`N = 1` is a disaster (−14.23 pp).** Holding ≥ 2 jokers in hand is
   reasonably common (full deck has 18 jokers in 162 cards, ~11%). At
   `N=1`, every such turn forces the bot to skip joker-using stock or
   hand-clear plays — which are exactly the highest-leverage moves
   Heuristic 13 makes. Games drag on (h13 alone takes +301k extra
   decisions) because the bot keeps stalling into discards.
2. **`N = 2` is mildly bad (−1.38 pp).** Holding ≥ 3 jokers is rarer, so
   suppression triggers less often, but when it does the bot is again
   passing up a strong combo. h13's excess decisions: +44k.
3. **`N ≥ 3` is indistinguishable from plain Heuristic 13.** Holding ≥ 4
   jokers in a 5-card hand is uncommon and the marginal "save vs spend"
   decision rarely matters; the win-rate hit dissolves into noise. By
   `N = 5` the suppression branch is unreachable (joker_count ≤ hand_size
   = 5), so the bot reproduces plain Heuristic 13 bit-for-bit.

So: jokers are most valuable when they are spent in the very combos
Heuristic 13 already builds — using them to bridge a stock-progression
prefix or to enable a full-hand-clear that triggers a redraw. There is
no observable upside to hoarding them past those moments. The intuition
behind hoarding ("save the wild card for an even bigger play") would
need a smarter trigger than "I happen to have ≥ N+1 of them in hand
right now" to overcome the cost of skipping the immediate combo.

## Caveats and possible follow-ups

- The "save" branch is blunt: it suppresses *any* joker play, including
  cases where the joker would have been spent on a small advance with
  modest value. A subtler bot might instead suppress only the costlier
  joker uses (e.g. only refuse to consume a joker when the build pile
  it's targeting is far from the stock value), which could move the
  curve at intermediate `N`.
- This is a 1 v 1 setup with `--stock-size 20`. Larger stocks or
  multi-player tables change the value of stock-progress combos, and
  could shift the cross-over point.
- We compare against pure Heuristic 13. A meta-bot opponent that
  exploits joker-hoarding (e.g. by aggressively running stock when
  `jokertactic` is known to be saving) might widen the gap further.
