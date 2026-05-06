use crate::action::Action;
use crate::bot::Bot;
use crate::bots::heuristic_13::Heuristic13Bot;
use crate::state::GameStateView;

/// Joker-refresh bot: extends the strongest existing heuristic (Heuristic 13,
/// "combo architect") with the deck-refresh mechanic.
///
/// Decision policy:
/// 1. Run Heuristic 13 to pick a candidate action.
/// 2. If the candidate is a Play, keep it — productive plays always beat
///    cycling the hand.
/// 3. Otherwise, if any `Action::Refresh { jokers_paid: k }` is legal, pick
///    the *smallest* legal `k` (= 1) and refresh. Each additional joker
///    spent only buys us swapping one more non-joker, and jokers are far
///    more valuable than the random card we'd get back, so wasting jokers
///    beyond what's needed to trigger the action is strictly bad.
/// 4. Otherwise fall back to Heuristic 13's choice (likely a Discard).
///
/// The refresh cap is configured at the game level
/// (`GameSettings.max_refresh_jokers`); the bot reads it indirectly via
/// `legal_actions`.
pub struct JokerRefreshBot {
    inner: Heuristic13Bot,
}

impl JokerRefreshBot {
    pub fn new() -> Self {
        Self {
            inner: Heuristic13Bot::new(),
        }
    }
}

impl Default for JokerRefreshBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for JokerRefreshBot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        let pick = self.inner.select_action(state, legal_actions);

        if matches!(pick, Action::Play { .. }) {
            return pick;
        }
        // Sanity gate: only refresh if there are enough cards in the deck +
        // recycle pile to actually refill what we'd remove.
        if state.draw_pile_count + state.recycle_pile_count == 0 {
            return pick;
        }
        let cheapest_k = legal_actions
            .iter()
            .filter_map(|a| match a {
                Action::Refresh { jokers_paid } => Some(*jokers_paid),
                _ => None,
            })
            .min();
        match cheapest_k {
            Some(k) => Action::Refresh { jokers_paid: k },
            None => pick,
        }
    }
}
