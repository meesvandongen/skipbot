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
- 10,000 games per scenario, seating permuted each game.
- `--stock-size 20`, `--max-turns 10000` (matches `heuristic_research.md`).
- Seed: default (`0xC0FFEE...5EED`), so all scenarios use identical decks
  and seating; only the refresh cost changes between rows.
- Command template:
  `cargo run --quiet --release --bin winrate -- --games 10000 --no-chart --max-turns 10000 --stock-size 20 --refresh-cost <N> jokerrefresh heuristic13`

## Results

| Scenario          | jokerrefresh win % | heuristic13 win % | Δ (jr − h13) | jr decisions | h13 decisions | Notes                                                              |
| ----------------- | ------------------ | ----------------- | -----------: | -----------: | ------------: | ------------------------------------------------------------------ |
| no refresh (ctrl) | 50.59              | 49.23             |        +1.36 |    1,037,348 |     1,031,639 | Sanity: refresh disabled, behavior identical to heuristic13.       |
| cost = 1 joker    | 43.95              | 55.99             |       −12.04 |    1,114,098 |     1,060,784 | Refresh fires constantly (~5–8×/game); jr collapses.               |
| cost = 2 jokers   | 46.75              | 53.03             |        −6.28 |    1,043,604 |     1,056,873 | Refresh still fires often and is value-destructive.                |
| cost = 3 jokers   | 50.51              | 49.28             |        +1.23 |    1,037,590 |     1,036,038 | Refresh rarely fires; jr ≈ heuristic13 within noise.               |
| cost = 4 jokers   | 50.60              | 49.22             |        +1.38 |    1,037,395 |     1,031,683 | Refresh effectively never fires (need 4/5 hand cards to be jokers). |
| cost = 5 jokers   | 50.59              | 49.23             |        +1.36 |    1,037,348 |     1,031,639 | Identical to "no refresh" scenario — never fires.                  |

(*"jr decisions" / "h13 decisions" are total `select_action` calls across
the 10,000 games. The deltas above the no-refresh baseline measure how
often `jokerrefresh` actually invoked the refresh action: each refresh
adds one extra decision for the active player on the same turn.*)

## Interpretation

The headline answer to "how many jokers should be traded for a deck
refresh?" using this naive trigger policy is: **none of them — at any of
the tested costs, the wrapper is at best neutral and at worst loses 12
percentage points of win rate.**

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
3. **At cost ≥ 3 the action becomes self-limiting.** Because a fresh
   hand size is 5, holding 3+ jokers concurrently is uncommon, and when
   it does happen heuristic13 has usually already played one of them
   for stock progress. The action almost never appears legal, so the
   policy regresses to pure heuristic13.
4. **Cost 5 is functionally a no-op.** Triggering it requires all five
   hand cards to be jokers; even if that happens, heuristic13 will
   normally play at least one of them onto a build pile before the
   wrapper gets a chance to trigger refresh.
5. **Determinism check.** Cost = 5 reproduces the no-refresh scenario
   bit-for-bit (same wins, same decision counts), confirming the
   refresh path is never taken in that regime.

## Caveats and possible follow-ups

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
