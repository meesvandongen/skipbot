use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::bots::heuristic_13::Heuristic13Bot;
use crate::card::Card;
use crate::state::{GameStateView, PlayerPublicState};

/// Joker-tactic bot: tests *when* a heuristic-13 player should be willing to
/// spend its Skip-Bo cards on the standard combo plays (stock-progression
/// prefix + full-hand-clear) versus saving them for a later turn.
///
/// Configured by `max_jokers_in_play: usize`. The policy is:
///
/// - Count the jokers in the current hand.
/// - If `joker_count == 0` or `joker_count <= max_jokers_in_play`, defer to
///   plain Heuristic 13. The "tactic" of using the jokers in this turn's
///   stock-clearing / hand-clearing combo is *allowed*.
/// - Otherwise (we hold *more* jokers than the budget allows us to spend in
///   one turn), forbid playing any joker this turn — strip every Play action
///   that would consume a Skip-Bo from the hand, the stock, or a discard top
///   from the legal-actions list, and re-run Heuristic 13 against the
///   filtered list. Heuristic 13 will fall through to a non-joker play, a
///   discard, or end-turn as appropriate.
///
/// The intent is to measure: when we have a number of jokers that exceeds
/// the budget `N`, is it better to hoard them (save for a later, possibly
/// more decisive, turn) than to spend any of them in the same hand-clear
/// combos that Heuristic 13 would normally pick? Different values of `N`
/// answer this for different "I have at most N jokers" thresholds.
///
/// Following the user's framing:
///
/// > if 1 is allowed we will only ever do this strategy if there is exactly
/// > 1 in hand. if 2 is allowed, we will do the strategy with 1 or 2 in hand.
pub struct JokerTacticBot {
    inner: Heuristic13Bot,
    max_jokers_in_play: usize,
}

impl JokerTacticBot {
    pub fn new(max_jokers_in_play: usize) -> Self {
        Self {
            inner: Heuristic13Bot::new(),
            max_jokers_in_play,
        }
    }

    fn self_player<'a>(state: &'a GameStateView) -> &'a PlayerPublicState {
        state
            .players
            .iter()
            .find(|p| p.id == state.self_player)
            .expect("self player must be present")
    }

    /// Returns true if `action` is a Play that consumes a Skip-Bo card from
    /// any source (hand, stock top, or discard top).
    fn plays_a_joker(state: &GameStateView, action: &Action) -> bool {
        let Action::Play { source, .. } = action else {
            return false;
        };
        match *source {
            CardSource::Hand(i) => matches!(state.hand.get(i), Some(Card::SkipBo)),
            CardSource::Stock => {
                matches!(Self::self_player(state).stock_top, Some(Card::SkipBo))
            }
            CardSource::Discard(d) => matches!(
                Self::self_player(state)
                    .discard_piles
                    .get(d)
                    .and_then(|pile| pile.last())
                    .copied(),
                Some(Card::SkipBo)
            ),
        }
    }
}

impl Default for JokerTacticBot {
    fn default() -> Self {
        // Default budget = 1 joker per turn. Matches the most conservative
        // setting in the matrix and yields predictable behavior when the
        // bot is created without a parameter.
        Self::new(1)
    }
}

impl Bot for JokerTacticBot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        let joker_count = state
            .hand
            .iter()
            .filter(|c| matches!(c, Card::SkipBo))
            .count();

        if joker_count == 0 || joker_count <= self.max_jokers_in_play {
            // Tactic allowed: defer to Heuristic 13 unmodified.
            return self.inner.select_action(state, legal_actions);
        }

        // joker_count > max: save the jokers. Suppress every Play action
        // that would consume one and re-run Heuristic 13.
        let filtered: Vec<Action> = legal_actions
            .iter()
            .filter(|a| !Self::plays_a_joker(state, a))
            .cloned()
            .collect();

        if filtered.is_empty() {
            // No non-joker action exists (very rare, but possible at
            // game-end states). Fall back to vanilla Heuristic 13.
            return self.inner.select_action(state, legal_actions);
        }
        self.inner.select_action(state, &filtered)
    }
}
