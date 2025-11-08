use serde::{Deserialize, Serialize};

use crate::action::PlayerId;
use crate::card::{BUILD_PILE_COUNT, Card, DISCARD_PILE_COUNT, HAND_SIZE, MAX_PLAYERS};
use crate::error::GameError;

/// Global constants for a running game.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GameSettings {
    pub num_players: usize,
    pub stock_size: usize,
    pub hand_size: usize,
    pub discard_piles: usize,
    pub build_piles: usize,
}

impl GameSettings {
    pub fn new(num_players: usize) -> Result<Self, GameError> {
        if !(2..=MAX_PLAYERS).contains(&num_players) {
            return Err(GameError::InvalidConfiguration(
                "players must be between 2 and 6",
            ));
        }
        let stock_size = if num_players <= 4 { 30 } else { 20 };
        Ok(Self {
            num_players,
            stock_size,
            hand_size: HAND_SIZE,
            discard_piles: DISCARD_PILE_COUNT,
            build_piles: BUILD_PILE_COUNT,
        })
    }
}

/// Public information regarding a build pile.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BuildPileView {
    pub cards: Vec<Card>,
    pub next_value: u8,
}

impl BuildPileView {
    pub fn empty() -> Self {
        Self {
            cards: Vec::new(),
            next_value: 1,
        }
    }
}

/// Public portion of a player's state that all opponents may observe.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlayerPublicState {
    pub id: PlayerId,
    pub stock_count: usize,
    pub stock_top: Option<Card>,
    pub discard_tops: [Option<Card>; DISCARD_PILE_COUNT],
    pub discard_counts: [usize; DISCARD_PILE_COUNT],
    pub hand_size: usize,
    pub is_current: bool,
    pub has_won: bool,
}

/// Status of the entire game.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum GameStatus {
    Ongoing,
    Finished { winner: PlayerId },
    Draw,
}

/// Current phase of the active turn.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum TurnPhase {
    AwaitingAction,
    GameOver,
}

/// Game state snapshot tailored for bots and ML agents.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct GameStateView {
    pub settings: GameSettings,
    pub phase: TurnPhase,
    pub status: GameStatus,
    pub self_player: PlayerId,
    pub current_player: PlayerId,
    pub draw_pile_count: usize,
    pub recycle_pile_count: usize,
    pub build_piles: [BuildPileView; BUILD_PILE_COUNT],
    pub players: Vec<PlayerPublicState>,
    pub hand: Vec<Card>,
}
