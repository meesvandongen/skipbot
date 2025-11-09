//! Skip-Bo game engine tailored for reinforcement learning workloads and bot experimentation.

pub mod action;
pub mod bot;
pub mod bots;
pub mod card;
pub mod error;
pub mod game;
pub mod ml;
pub mod score;
pub mod state;
pub mod visualize;

pub use crate::action::{Action, CardSource};
pub use crate::bot::Bot;
pub use crate::bots::{Heuristic2Bot, HeuristicBot, HumanBot, PolicyBot, RandomBot};
pub use crate::bots::{create_bot_from_spec, label_for_spec, parse_policy_spec};
pub use crate::card::Card;
pub use crate::error::{GameError, InvalidAction};
pub use crate::game::{Game, GameBuilder, GameConfig};
pub use crate::ml::{
    ActionSpace, DEFAULT_HIDDEN, DEFAULT_OUTPUT, DEFAULT_STACK, PolicyNetwork, StateEncoder,
};
pub use crate::score::winner_points;
pub use crate::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};
pub use crate::visualize::{DescribeOptions, VisualOptions, describe_action, render_state};
