use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::state::GameStateView;

/// Heuristic 15 bot: variant of Heuristic 14 that prioritizes plays in order:
/// Stock > Hand > Discard. Otherwise behaves the same (always play if possible).
/// Policy:
/// 1. If any Stock play is legal, choose it.
/// 2. Else any Hand play.
/// 3. Else any Discard play.
/// 4. Else EndTurn if legal.
/// 5. Else first Discard (fallback) or first action.
pub struct Heuristic15Bot;

impl Heuristic15Bot {
    pub fn new() -> Self {
        Self
    }
}
impl Default for Heuristic15Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic15Bot {
    fn select_action(&mut self, _state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 15 bot requires at least one legal action"
        );
        if let Some(a) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Stock,
                    ..
                }
            )
        }) {
            return a.clone();
        }
        if let Some(a) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Hand(_),
                    ..
                }
            )
        }) {
            return a.clone();
        }
        if let Some(a) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Discard(_),
                    ..
                }
            )
        }) {
            return a.clone();
        }
        if let Some(a) = legal_actions.iter().find(|a| matches!(a, Action::EndTurn)) {
            return a.clone();
        }
        if let Some(a) = legal_actions
            .iter()
            .find(|a| matches!(a, Action::Discard { .. }))
        {
            return a.clone();
        }
        legal_actions[0].clone()
    }
}
