use crate::action::Action;
use crate::state::GameStateView;

/// Interface for defining custom Skip-Bo bots.
pub trait Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action;
}
