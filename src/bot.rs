use std::io::{self, Write};

use burn::tensor::backend::Backend;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::action::{Action, CardSource};
use crate::card::{Card, MAX_CARD_VALUE};
use crate::ml::{ActionSpace, PolicyNetwork, StateEncoder};
use crate::state::{GameStateView, PlayerPublicState};
use crate::visualize::{describe_action, render_state};

/// Interface for defining custom Skip-Bo bots.
pub trait Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action;
}

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

