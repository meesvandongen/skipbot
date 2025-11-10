use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::state::GameStateView;

/// Heuristic 14 bot: naive player that always plays whenever possible.
///
/// Policy:
/// - If any play is legal this turn, take a play action.
///   Preference order among plays: Stock > Discard > Hand > otherwise first Play.
/// - Otherwise, if EndTurn is legal (hand empty), end the turn.
/// - Otherwise, discard arbitrarily (first available discard action).
pub struct Heuristic14Bot;

impl Heuristic14Bot {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Heuristic14Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic14Bot {
    fn select_action(&mut self, _state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 14 bot requires at least one legal action"
        );

        // 1) Prefer any Play action to keep the turn going; choose naive priority.
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Stock,
                    ..
                }
            )
        }) {
            return action.clone();
        }
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Discard(_),
                    ..
                }
            )
        }) {
            return action.clone();
        }
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Hand(_),
                    ..
                }
            )
        }) {
            return action.clone();
        }

        // 2) If we can end the turn, do so.
        if let Some(action) = legal_actions.iter().find(|a| matches!(a, Action::EndTurn)) {
            return action.clone();
        }

        // 3) Otherwise discard (naively pick the first discard available) or fallback.
        if let Some(action) = legal_actions
            .iter()
            .find(|a| matches!(a, Action::Discard { .. }))
        {
            return action.clone();
        }

        // Fallback: return the first legal action to remain robust.
        legal_actions[0].clone()
    }
}
