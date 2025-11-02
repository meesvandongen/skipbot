use rand::seq::SliceRandom;
use rand::Rng;

use crate::action::Action;
use crate::bot::Bot;
use crate::state::GameStateView;

/// Baseline bot that samples uniformly from the legal action set.
pub struct RandomBot<R: Rng> {
    rng: R,
}

impl<R: Rng> RandomBot<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Bot for RandomBot<R> {
    fn select_action(&mut self, _state: &GameStateView, legal_actions: &[Action]) -> Action {
        legal_actions
            .choose(&mut self.rng)
            .cloned()
            .expect("at least one legal action must be available")
    }
}

