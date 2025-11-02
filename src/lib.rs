//! Skip-Bo game engine tailored for reinforcement learning workloads and bot experimentation.

pub mod action;
pub mod bot;
pub mod card;
pub mod error;
pub mod game;
pub mod ml;
pub mod state;
pub mod visualize;

pub use crate::action::{Action, CardSource};
pub use crate::bot::{Bot, HeuristicBot, HumanBot, PolicyBot, RandomBot};
pub use crate::card::Card;
pub use crate::error::{GameError, InvalidAction};
pub use crate::game::{Game, GameBuilder, GameConfig};
pub use crate::ml::{
    ActionSpace, DEFAULT_HIDDEN, DEFAULT_OUTPUT, DEFAULT_STACK, PolicyNetwork, StateEncoder,
};
pub use crate::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};
pub use crate::visualize::{DescribeOptions, VisualOptions, describe_action, render_state};
