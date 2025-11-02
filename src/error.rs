use thiserror::Error;

use crate::action::PlayerId;

/// Errors that can occur when manipulating the game state.
#[derive(Debug, Error)]
pub enum GameError {
    #[error("player index {0} is out of range")]
    InvalidPlayer(PlayerId),
    #[error("not the specified player's turn")]
    NotPlayersTurn,
    #[error("invalid action: {0}")]
    InvalidAction(#[from] InvalidAction),
    #[error("game is already over")]
    GameOver,
    #[error("invalid configuration: {0}")]
    InvalidConfiguration(&'static str),
}

/// Details of invalid user actions.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum InvalidAction {
    #[error("hand index {0} is out of range")]
    HandIndex(usize),
    #[error("discard pile index {0} is out of range")]
    DiscardIndex(usize),
    #[error("build pile index {0} is out of range")]
    BuildPileIndex(usize),
    #[error("no card available in the selected source")]
    NoCardAvailable,
    #[error("card does not match required value {required}")]
    CardMismatch { required: u8 },
    #[error("player must discard before ending turn")]
    MustDiscard,
    #[error("player cannot discard because hand is empty")]
    EmptyHand,
}
