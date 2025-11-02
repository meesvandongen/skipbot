use burn::tensor::backend::Backend;

use crate::action::Action;
use crate::bot::Bot;
use crate::ml::{ActionSpace, PolicyNetwork, StateEncoder};
use crate::state::GameStateView;

/// Policy-driven bot backed by a Burn neural network.
pub struct PolicyBot<B: Backend> {
    policy: PolicyNetwork<B>,
}

impl<B: Backend> PolicyBot<B> {
    pub fn new(policy: PolicyNetwork<B>) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &PolicyNetwork<B> {
        &self.policy
    }
}

impl<B: Backend> Bot for PolicyBot<B> {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "policy bot requires at least one legal action"
        );
        let input = StateEncoder::encode_tensor::<B>(state);
        let logits = self.policy.forward(input).reshape([ActionSpace::MAX]);
        let values: Vec<f32> = logits
            .into_data()
            .to_vec::<f32>()
            .expect("tensor conversion");
        let mut best: Option<(f32, Action)> = None;
        for action in legal_actions {
            if let Some(index) = ActionSpace::action_index(action) {
                let value = values[index];
                match &mut best {
                    Some((best_value, best_action)) => {
                        if value > *best_value {
                            *best_value = value;
                            *best_action = action.clone();
                        }
                    }
                    None => best = Some((value, action.clone())),
                }
            }
        }
        best.map(|(_, action)| action)
            .unwrap_or_else(|| legal_actions[0].clone())
    }
}
