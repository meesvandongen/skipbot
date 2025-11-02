use burn_ndarray::NdArray;

use skipbot::ml::{ActionSpace, PolicyNetwork, STATE_FEATURES, StateEncoder};
use skipbot::{Bot, GameBuilder, PolicyBot};
type Backend = NdArray<f32>;

#[test]
fn encoder_outputs_expected_length() {
    let game = GameBuilder::new(2).expect("builder").build().expect("game");
    let view = game.state_view(0).expect("state view");
    let encoded = StateEncoder::encode(&view);
    assert_eq!(encoded.len(), STATE_FEATURES);
}

#[test]
fn policy_network_and_bot_return_legal_action() {
    let game = GameBuilder::new(2).expect("builder").build().expect("game");
    let view = game.state_view(0).expect("state view");
    let mut bot = PolicyBot::<Backend>::new(PolicyNetwork::<Backend>::default());
    let legal_actions = game.legal_actions(0).expect("legal actions");
    let action = bot.select_action(&view, &legal_actions);
    assert!(legal_actions.contains(&action));
    let index = ActionSpace::action_index(&action).expect("mapped index");
    assert!(index < ActionSpace::MAX);
}
