use crate::action::Action;
use crate::bot::Bot;
use crate::bots::heuristic_13::Heuristic13Bot;
use crate::card::Card;
use crate::state::GameStateView;

/// Joker-refresh bot: extends the strongest existing heuristic (Heuristic 13,
/// "combo architect") with the deck-refresh mechanic.
///
/// Decision policy:
/// 1. Run Heuristic 13 to pick a candidate action.
/// 2. If the candidate is a Play, keep it — productive plays always beat
///    burning the hand.
/// 3. Otherwise, if `Action::Refresh` is legal AND the hand currently contains
///    at least one non-joker card AND there are still cards to draw from, swap
///    the candidate for `Action::Refresh`. The intent is "rather than discard
///    a useless numbered card to a personal pile and end the turn empty-handed,
///    spend the jokers I'm not using to cycle the hand for another shot."
/// 4. Otherwise fall back to Heuristic 13's choice.
///
/// The refresh cost is configured at the game level
/// (`GameSettings.refresh_cost`); the bot inspects it but does not own it.
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
        if !legal_actions.contains(&Action::Refresh) {
            return pick;
        }
        // Cycling makes sense only if we have something other than the cost
        // tokens to swap out, and there is still a deck to draw from.
        let non_joker = state
            .hand
            .iter()
            .filter(|c| !matches!(c, Card::SkipBo))
            .count();
        if non_joker == 0 {
            return pick;
        }
        if state.draw_pile_count + state.recycle_pile_count == 0 {
            return pick;
        }
        Action::Refresh
    }
}
