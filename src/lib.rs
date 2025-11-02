//! Skip-Bo game engine tailored for reinforcement learning workloads and bot experimentation.

pub mod action;
pub mod bot;
pub mod card;
pub mod error;
pub mod game;
pub mod state;
pub mod visualize;

pub use crate::action::{Action, CardSource};
pub use crate::bot::{Bot, HumanBot, RandomBot};
pub use crate::card::Card;
pub use crate::error::{GameError, InvalidAction};
pub use crate::game::{Game, GameBuilder, GameConfig};
pub use crate::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};
pub use crate::visualize::{DescribeOptions, VisualOptions, describe_action, render_state};
