# Joker Refresh Experiment

How many jokers (Skip-Bo wild cards) should be traded for a deck refresh?

## Mechanic

The engine gains an optional rule, configured via
`GameBuilder::with_max_refresh_jokers(max)` (or `--max-refresh-jokers` on
the `winrate` binary). When the rule is enabled, players gain access to
a parameterised action `Action::Refresh { jokers_paid: k }` that:

1. Is legal for any `k ∈ [1, max]` such that the active player holds
   **at least `k` Skip-Bo cards AND at least `k` non-Skip-Bo cards** in
   their hand.
2. Sends `k` Skip-Bo cards (the cost) and `k` non-joker cards (the
   benefit — the "stuck" cards being cycled out) into the recycle pile.
3. Draws `2k` fresh cards back into the hand, keeping the hand size at
   `hand_size = 5`.
4. Does **not** end the turn — the player continues with their refreshed
   hand.

This is the "up-to-N jokers" semantics: `max` is a *cap*, not a fixed
price. The player chooses any `k` in `[1, max]` per use, so a higher
`max` strictly subsumes a lower one (more options for the player).

Vanilla Skip-Bo rules are preserved when `max_refresh_jokers = None`.

> Note on the previous design. An earlier iteration of this experiment
> used a single fixed cost: `Action::Refresh` always burned the entire
> hand for exactly `cost` Skip-Bos, and only became legal when the
> player held ≥`cost` jokers. That conflated per-use price with the
> eligibility threshold and produced a non-monotonic curve. The current
> design separates the two.

## Bot Policy: `jokerrefresh`

The `jokerrefresh` bot wraps the strongest existing heuristic
(`heuristic13`, "combo architect", 95.71% baseline win rate). It
delegates to heuristic13 for all play decisions, then:

1. If heuristic13 returns a `Play`, keep it — productive plays always
   beat cycling cards.
2. If at least one `Action::Refresh { jokers_paid: k }` is legal, pick
   the **smallest** legal `k` (= 1) and refresh. Each additional joker
   spent only buys swapping one more non-joker card, and jokers are far
   more valuable than the random card we'd get back, so wasting jokers
   beyond what's needed to trigger the action is strictly bad.
3. Otherwise fall back to heuristic13's choice (typically a `Discard`).

Because the bot always picks k = 1, raising `max_refresh_jokers` above
1 does not change its behaviour. That is in fact the whole point of
this experiment: it confirms that the "up-to-N" reading of cost makes
the rule monotonic in `max` (more options never hurt).

## Methodology

- Match-up: `jokerrefresh` (under test) vs. `heuristic13` (control), 1 v 1.
- 100,000 games per scenario, seating permuted each game.
- `--stock-size 20`, `--max-turns 10000` (matches `heuristic_research.md`).
- Seed: default (`0xC0FFEE...5EED`), so all scenarios use identical
  decks and seating.
- At 100,000 games per cell the standard error of the win-rate estimate
  is ≈ √(0.25 / 100000) ≈ 0.16 pp; differences smaller than ~0.5 pp
  are noise.
- Command template:
  `cargo run --quiet --release --bin winrate -- --games 100000 --no-chart --max-turns 10000 --stock-size 20 --max-refresh-jokers <N> jokerrefresh heuristic13`

## Results

| Scenario           | jokerrefresh win % | heuristic13 win % | Δ (jr − h13) |  jr decisions | h13 decisions | Notes                                                                |
| ------------------ | -----------------: | ----------------: | -----------: | ------------: | ------------: | -------------------------------------------------------------------- |
| no refresh (ctrl)  |              50.11 |             49.72 |       +0.39 |    10,324,972 |    10,328,169 | Sanity: refresh disabled, behaviour identical to heuristic13.         |
| max = 1 joker      |              33.59 |             66.27 |      −32.68 |    11,275,015 |    11,453,628 | Refresh fires every time the bot is stuck and has ≥1 joker + ≥1 non-joker. |
| max = 2 jokers     |              33.59 |             66.27 |      −32.68 |    11,275,015 |    11,453,628 | Bit-identical to max=1 — bot picks k=1 in every case.                  |
| max = 3 jokers     |              33.59 |             66.27 |      −32.68 |    11,275,015 |    11,453,628 | Bit-identical.                                                          |
| max = 4 jokers     |              33.59 |             66.27 |      −32.68 |    11,275,015 |    11,453,628 | Bit-identical.                                                          |
| max = 5 jokers     |              33.59 |             66.27 |      −32.68 |    11,275,015 |    11,453,628 | Bit-identical.                                                          |

(*"jr decisions" / "h13 decisions" are total `select_action` calls
across the 100,000 games. `jokerrefresh` accumulates +950k decisions
over its no-refresh baseline (10.32M → 11.28M); `heuristic13`
accumulates +1.13M decisions over its baseline (10.33M → 11.45M)
because games run longer when jokerrefresh is bleeding stock.*)

## Interpretation

Two clean facts emerge from this run:

1. **The curve is flat in `max`.** All five scenarios produce
   bit-identical numbers — same wins, same decisions, same RNG
   trajectory. That is the property that the original
   ("must pay exactly N") design failed: under the "up-to-N" semantics,
   a higher cap is a strict superset of options for the player, and
   since the bot's optimal choice is k = 1, more options simply go
   unused.
2. **The refresh action itself is value-destructive at every cap.**
   The bot loses ~32 pp of win rate against heuristic13 by ever using
   refresh, even at k = 1 (the minimum-cost variant). That's worse
   than the previous experiment's cost = 1 (−12 pp). The reason is
   that the smaller per-use effect (swap 2 cards instead of burning
   the whole hand) doesn't unstick the bot — it discards 1 of its
   "useless" non-jokers, draws 2 random cards, often gets stuck
   again, refreshes again, and so on. Excess `jokerrefresh` decisions
   over its no-refresh baseline went from +777k under the old "burn
   whole hand" design to +950k here, and total decisions across both
   bots grew by ~10% (games run longer when jokerrefresh is losing
   stock parity).

Putting (1) and (2) together: under this naive stuck-trigger policy,
**no value of `max ≥ 1` is worth using — they are all equally bad,
and they are bad because the bot uses them at all.** The flatness
result confirms the user's intuition that "up-to-N" eliminates the
non-monotonicity artefact of the previous design; the level of the
flat line confirms that even k = 1 is an unfavourable trade for the
*current* bot.

## Why this is consistent with the earlier experiment

The earlier "fixed cost" run measured cost = 1 at −12.29 pp; this run
measures the analogous k = 1 at −32.68 pp. Both use the same trigger
condition ("stuck + ≥1 joker"), so frequency-of-use is similar, but
the per-use effect differs:

- **Old cost = 1**: pay 1 joker, burn the entire 5-card hand, redraw
  5. Big per-use effect; one refresh resets you completely.
- **New k = 1**: pay 1 joker + cycle 1 non-joker, redraw 2. Small
  per-use effect; the rest of the hand still has the cards that made
  you stuck, so you re-trigger immediately.

So the new mechanic is *cheaper per use* but *less effective per use*,
and the bot ends up using it more often per game (excess jr decisions
grew from +777k to +950k). The win-rate hit grows correspondingly.

## Caveats and possible follow-ups

- **The bot is still naive.** It refreshes on every stuck turn. A
  smarter policy that decides *whether* to refresh (rather than
  always doing so when legal) could bring the result toward the
  no-refresh baseline. The current run only establishes that the
  trigger-when-stuck heuristic is wrong; it does not establish that
  the mechanic is unwinnable.
- **The engine picks which non-jokers to swap (leftmost first).**
  Letting the bot pick the worst non-jokers (e.g. the ones with no
  near-future plays) would make each refresh more useful per joker
  spent. That extension widens the action space and is left as a
  follow-up.
- **All games use 1 v 1, `--stock-size 20`, `--max-turns 10000`.**
  Larger stocks, multi-player tables, or different turn limits could
  shift the value of cycling.
