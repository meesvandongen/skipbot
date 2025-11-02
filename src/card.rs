use serde::{Deserialize, Serialize};

/// Representation of a Skip-Bo card.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub enum Card {
    /// Numbered card between 1 and 12.
    Number(u8),
    /// Skip-Bo wild card. Counts as any number when played.
    SkipBo,
}

pub const MIN_CARD_VALUE: u8 = 1;
pub const MAX_CARD_VALUE: u8 = 12;
pub const SKIP_BO_COUNT: usize = 18;
pub const COPIES_PER_VALUE: usize = 12;
pub const HAND_SIZE: usize = 5;
pub const DISCARD_PILE_COUNT: usize = 4;
pub const BUILD_PILE_COUNT: usize = 4;
pub const MAX_PLAYERS: usize = 6;

impl Card {
    /// Returns true if the card is the wildcard.
    #[inline]
    pub fn is_skip_bo(&self) -> bool {
        matches!(self, Card::SkipBo)
    }

    /// Returns the numeric value when available.
    #[inline]
    pub fn value(&self) -> Option<u8> {
        match self {
            Card::Number(v) => Some(*v),
            Card::SkipBo => None,
        }
    }

    /// Checks whether the card can legally satisfy the requested value.
    #[inline]
    pub fn matches_value(&self, value: u8) -> bool {
        debug_assert!(value >= MIN_CARD_VALUE && value <= MAX_CARD_VALUE);
        matches!(self, Card::SkipBo) || self.value() == Some(value)
    }
}

/// Builds a full 162-card Skip-Bo deck in deterministic order (unshuffled).
pub fn full_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(162);
    for _ in 0..COPIES_PER_VALUE {
        for value in MIN_CARD_VALUE..=MAX_CARD_VALUE {
            deck.push(Card::Number(value));
        }
    }
    deck.extend(std::iter::repeat(Card::SkipBo).take(SKIP_BO_COUNT));
    deck
}
