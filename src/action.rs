use serde::{Deserialize, Serialize};

use crate::card::BUILD_PILE_COUNT;

/// Zero-based index of a player within the game.
pub type PlayerId = usize;

/// Location a card can be taken from when performing a play action.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum CardSource {
    /// Card taken from the active player's hand by index.
    Hand(usize),
    /// Card taken from the active player's stock pile (top card).
    Stock,
    /// Card taken from one of the active player's discard piles (top card).
    Discard(usize),
}

/// Action available to an agent during its turn.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Action {
    /// Play a card from a source onto the specified build pile.
    Play {
        source: CardSource,
        build_pile: usize,
    },
    /// Discard a card from the hand onto one of the four personal discard piles.
    Discard {
        hand_index: usize,
        discard_pile: usize,
    },
    /// Finish the turn when the hand is empty.
    EndTurn,
}

impl Action {
    /// Returns the build pile index if the action is a play.
    pub fn build_pile(&self) -> Option<usize> {
        match self {
            Action::Play { build_pile, .. } => Some(*build_pile),
            _ => None,
        }
    }

    /// Validates whether the build pile index lies within range.
    pub fn build_pile_in_range(&self) -> bool {
        self.build_pile()
            .map(|idx| idx < BUILD_PILE_COUNT)
            .unwrap_or(true)
    }
}
