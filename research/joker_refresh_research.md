# Joker Refresh Experiment

How many jokers (Skip-Bo wild cards) should be traded for a deck refresh?

## Mechanic

The engine gains a new optional rule, configured via
`GameBuilder::with_refresh_cost(cost)` (or `--refresh-cost` on the
`winrate` binary). When the rule is enabled, players gain access to a new
`Action::Refresh` that:

1. Is legal only when the active player holds at least `cost` Skip-Bo cards
   in their hand and the hand is non-empty.
2. Sends the player's entire hand (the `cost` jokers paid plus any other
   cards) to the recycle pile.
3. Refills the player's hand back up to `hand_size = 5` from the draw pile
   (auto-recycling as usual when the draw pile is empty).
4. Does **not** end the turn — the player continues with their fresh hand.

Vanilla Skip-Bo rules are preserved when the cost is left unset
(`refresh_cost = None`).

## Bot Policy: `jokerrefresh`

A new bot, `jokerrefresh`, wraps the strongest existing heuristic
(`heuristic13`, "combo architect", 95.71% baseline win rate). It defers to
heuristic13 for all play decisions. When heuristic13 is about to discard
(i.e. no productive play exists this turn), the wrapper substitutes
`Action::Refresh` instead — but only when:

- The refresh action is legal (so enough jokers are in hand).
- The hand contains at least one non-joker card (otherwise we'd just be
  burning jokers for random replacements).
- The combined draw + recycle pool is non-empty (otherwise the redraw
  yields nothing).

Effectively: "rather than dump a useless numbered card to a personal pile,
spend the jokers I'm not currently using to cycle the hand for another
shot."

When `refresh_cost = None`, `Action::Refresh` is never legal, so
`jokerrefresh` collapses to plain heuristic13 — confirmed by the sanity
run below.

## Methodology

- Match-up: `jokerrefresh` (under test) vs. `heuristic13` (control), 1 v 1.
- 100,000 games per scenario, seating permuted each game.
- `--stock-size 20`, `--max-turns 10000` (matches `heuristic_research.md`).
- Seed: default (`0xC0FFEE...5EED`), so all scenarios use identical decks
  and seating; only the refresh cost changes between rows.
- Command template:
  `cargo run --quiet --release --bin winrate -- --games 100000 --no-chart --max-turns 10000 --stock-size 20 --refresh-cost <N> jokerrefresh heuristic13`
- At 100,000 games per cell the standard error of the win-rate estimate is
  ≈ √(0.25 / 100000) ≈ 0.16 pp; differences smaller than ~0.5 pp are noise.

## Results

| Scenario          | jokerrefresh win % | heuristic13 win % | Δ (jr − h13) | jr decisions | h13 decisions | Notes                                                                                                  |
| ----------------- | -----------------: | ----------------: | -----------: | -----------: | ------------: | ------------------------------------------------------------------------------------------------------ |
| no refresh (ctrl) |              50.11 |             49.72 |       +0.39 |  10,324,972 |   10,328,169 | Sanity: refresh disabled, behavior identical to heuristic13.                                            |
| cost = 1 joker    |              43.83 |             56.12 |      −12.29 |  11,102,281 |   10,618,668 | Refresh fires on most "stuck" turns; jr decisions +777k over baseline (~10× more refreshes than cost=2). |
| cost = 2 jokers   |              46.19 |             53.61 |       −7.42 |  10,386,975 |   10,586,833 | Refresh fires occasionally (jr decisions +62k over baseline); per-use cost higher but ~12× rarer.       |
| cost = 3 jokers   |              49.64 |             50.17 |       −0.53 |  10,319,518 |   10,369,636 | Refresh rarely fires; jr is mildly worse than the no-refresh control.                                  |
| cost = 4 jokers   |              50.09 |             49.74 |       +0.35 |  10,324,988 |   10,329,622 | Refresh effectively never fires (need ≥4/5 hand cards to be jokers).                                  |
| cost = 5 jokers   |              50.11 |             49.72 |       +0.39 |  10,324,972 |   10,328,169 | Bit-identical to no-refresh — never fires.                                                              |

(*"jr decisions" / "h13 decisions" are total `select_action` calls across
the 100,000 games. The excess of `jokerrefresh` decisions over the
no-refresh baseline (10.32M) is a frequency proxy: each refresh adds
several extra decisions on the same turn — try-to-play each of the 5
fresh cards.*)

## Interpretation

The headline answer to "how many jokers should be traded for a deck
refresh?" using this naive trigger policy is: **none of them — at every
tested cost the wrapper is no better than plain heuristic13, and at
costs 1–2 it loses 7–12 pp of win rate.**

A few observations explain why:

1. **Skip-Bo cards are the most valuable cards in the deck.** Jokers
   double as "skip a step" tokens during stock-clearing combos and as
   guaranteed plays on any build pile. Trading them away for a randomly
   drawn replacement loses expected stock-clearing throughput.
2. **Refresh fires only when the bot is "stuck."** That correlates with
   states where the hand is mostly small numbers, with a couple of
   jokers held in reserve. Cycling that hand burns the reserve to retry
   for the same shape of cards; the expected gain is small while the
   expected joker loss is the entire cost.
3. **The win-rate curve is non-monotonic in cost — and that's the
   frequency effect, not noise.** A natural intuition is that "cost=2
   should be at least as bad as cost=1, since you pay more jokers per
   refresh." But in this engine the cost parameter does double duty: it
   sets both the per-use price *and* the eligibility threshold (need
   ≥cost jokers in hand). Raising it from 1 to 2 cut refresh frequency
   by ~12× (jr-decision excess over baseline: +777k → +62k), which
   dwarfs the per-use-cost increase. Cost=1 is the worst because it's
   the cost at which the bot can — and does — abuse the action almost
   every stuck turn. By cost=3 the eligibility precondition is rare
   enough that the action is essentially gated off and the bot reverts
   to heuristic13.
4. **At cost ≥ 4 the action becomes self-limiting.** Holding 4+ jokers
   concurrently is uncommon, and when it happens heuristic13 has
   usually already played one of them for stock progress. The action
   almost never appears legal, so the policy regresses to pure
   heuristic13.
5. **Cost 5 is a determinism check.** It reproduces the no-refresh
   scenario bit-for-bit (50.11 vs 49.72, identical decision counts),
   confirming the refresh path is never taken in that regime.

## Caveats and possible follow-ups

- The cost parameter conflates eligibility-threshold and per-use-price.
  A cleaner experiment would decouple them: e.g. fix eligibility at
  ≥1 joker, and vary how many jokers are spent per refresh. That would
  isolate per-use cost from frequency-of-abuse and likely produce the
  monotonic curve the intuition predicts.
- This is a single bot-policy data point, not a search over policies. A
  smarter policy that refreshes only when the *specific* hand composition
  guarantees a stalled turn (e.g. all numbered cards above the current
  build-pile next-values, with no near-future stock-progress option) could
  in principle make refresh net-positive even at cost 1. The current
  result establishes only that the simple stuck-trigger policy is bad.
- All games use a 1 v 1 setup with `--stock-size 20`. Larger stocks or
  multi-player tables shift the value of cycling because turns between
  redraws are longer; results may differ.
- The mechanic itself is simple ("burn whole hand, redraw 5"). Variants
  worth exploring include keeping non-paid hand cards, only redrawing
  the slots paid, or letting refresh end the turn (which would make low
  costs much more competitive by limiting abuse).
