use burn::tensor::{Tensor, TensorData, backend::Backend};

use crate::action::{Action, CardSource};
use crate::card::{
    BUILD_PILE_COUNT, Card, DISCARD_PILE_COUNT, HAND_SIZE, MAX_CARD_VALUE, MAX_PLAYERS,
};
use crate::state::GameStateView;

pub const CARD_BUCKETS: usize = MAX_CARD_VALUE as usize + 1; // 1..12 + Skip-Bo
pub const BUILD_FEATURES: usize = BUILD_PILE_COUNT * 2;
pub const CURRENT_PLAYER_FEATURES: usize = 3;
pub const HAND_FEATURES: usize = CARD_BUCKETS;
pub const DISCARD_FEATURES: usize = DISCARD_PILE_COUNT * (CARD_BUCKETS + 1);
pub const PLAYER_FEATURES: usize = MAX_PLAYERS * 3;
pub const STATE_FEATURES: usize =
    BUILD_FEATURES + CURRENT_PLAYER_FEATURES + HAND_FEATURES + DISCARD_FEATURES + PLAYER_FEATURES;

pub const HAND_PLAY_ACTIONS: usize = HAND_SIZE * BUILD_PILE_COUNT;
pub const STOCK_PLAY_ACTIONS: usize = BUILD_PILE_COUNT;
pub const DISCARD_PLAY_ACTIONS: usize = DISCARD_PILE_COUNT * BUILD_PILE_COUNT;
pub const DISCARD_ACTIONS: usize = HAND_SIZE * DISCARD_PILE_COUNT;
pub const END_TURN_ACTIONS: usize = 1;

pub const HAND_PLAY_OFFSET: usize = 0;
pub const STOCK_PLAY_OFFSET: usize = HAND_PLAY_OFFSET + HAND_PLAY_ACTIONS;
pub const DISCARD_PLAY_OFFSET: usize = STOCK_PLAY_OFFSET + STOCK_PLAY_ACTIONS;
pub const DISCARD_OFFSET: usize = DISCARD_PLAY_OFFSET + DISCARD_PLAY_ACTIONS;
pub const END_TURN_INDEX: usize = DISCARD_OFFSET + DISCARD_ACTIONS;
pub const MAX_ACTIONS: usize = END_TURN_INDEX + END_TURN_ACTIONS;

#[inline]
fn normalize(value: usize, max: usize) -> f32 {
    if max == 0 {
        0.0
    } else {
        value as f32 / max as f32
    }
}

#[inline]
fn card_bucket(card: Card) -> usize {
    match card {
        Card::Number(value) => (value.saturating_sub(1)) as usize,
        Card::SkipBo => CARD_BUCKETS - 1,
    }
}

#[inline]
fn card_scalar(card: Card) -> f32 {
    card_bucket(card) as f32 / (CARD_BUCKETS - 1) as f32
}

pub struct StateEncoder;

impl StateEncoder {
    pub fn encode(state: &GameStateView) -> [f32; STATE_FEATURES] {
        let mut out = [0.0; STATE_FEATURES];
        let mut offset = 0;

        for pile in state.build_piles.iter() {
            out[offset] = normalize(pile.next_value as usize, MAX_CARD_VALUE as usize);
            offset += 1;
            out[offset] = normalize(pile.cards.len(), MAX_CARD_VALUE as usize);
            offset += 1;
        }

        let self_player = &state.players[state.self_player];
        out[offset] = normalize(self_player.stock_count, state.settings.stock_size);
        offset += 1;
        out[offset] = self_player.stock_top.map(card_scalar).unwrap_or(0.0);
        offset += 1;
        out[offset] = if self_player.has_won { 1.0 } else { 0.0 };
        offset += 1;

        let mut hand_counts = [0.0f32; CARD_BUCKETS];
        for card in &state.hand {
            let idx = card_bucket(*card);
            hand_counts[idx] += 1.0;
        }
        let hand_total = state.hand.len().max(1) as f32;
        for value in hand_counts {
            out[offset] = value / hand_total;
            offset += 1;
        }

        for discard_index in 0..state.settings.discard_piles {
            let mut bucket = [0.0f32; CARD_BUCKETS];
            if let Some(card) = self_player.discard_tops[discard_index] {
                bucket[card_bucket(card)] = 1.0;
            }
            for value in bucket {
                out[offset] = value;
                offset += 1;
            }
            out[offset] = normalize(
                self_player.discard_counts[discard_index],
                state.settings.stock_size,
            );
            offset += 1;
        }

        for player_index in 0..MAX_PLAYERS {
            let player = state.players.iter().find(|p| p.id == player_index);
            if let Some(player) = player {
                out[offset] = normalize(player.stock_count, state.settings.stock_size);
                offset += 1;
                let hand_size = if player.id == state.self_player {
                    state.hand.len()
                } else {
                    player.hand_size
                };
                out[offset] = normalize(hand_size, state.settings.hand_size);
                offset += 1;
                out[offset] = if player.has_won { 1.0 } else { 0.0 };
                offset += 1;
            } else {
                out[offset] = 0.0;
                offset += 1;
                out[offset] = 0.0;
                offset += 1;
                out[offset] = 0.0;
                offset += 1;
            }
        }

        debug_assert_eq!(offset, STATE_FEATURES);
        out
    }

    pub fn encode_tensor<B>(state: &GameStateView) -> Tensor<B, 2>
    where
        B: Backend,
        B::Device: Default,
    {
        let features = Self::encode(state);
        let data = TensorData::from([features]);
        Tensor::<B, 2>::from_data(data, &B::Device::default())
    }
}

pub struct ActionSpace;

impl ActionSpace {
    pub const MAX: usize = MAX_ACTIONS;

    pub fn action_index(action: &Action) -> Option<usize> {
        match action {
            Action::Play { source, build_pile } => {
                if *build_pile >= BUILD_PILE_COUNT {
                    return None;
                }
                match source {
                    CardSource::Hand(hand_index) => {
                        if *hand_index >= HAND_SIZE {
                            None
                        } else {
                            Some(HAND_PLAY_OFFSET + hand_index * BUILD_PILE_COUNT + build_pile)
                        }
                    }
                    CardSource::Stock => Some(STOCK_PLAY_OFFSET + build_pile),
                    CardSource::Discard(discard_index) => {
                        if *discard_index >= DISCARD_PILE_COUNT {
                            None
                        } else {
                            let relative = discard_index * BUILD_PILE_COUNT + build_pile;
                            Some(DISCARD_PLAY_OFFSET + relative)
                        }
                    }
                }
            }
            Action::Discard {
                hand_index,
                discard_pile,
            } => {
                if *hand_index >= HAND_SIZE || *discard_pile >= DISCARD_PILE_COUNT {
                    None
                } else {
                    Some(DISCARD_OFFSET + hand_index * DISCARD_PILE_COUNT + discard_pile)
                }
            }
            Action::EndTurn => Some(END_TURN_INDEX),
        }
    }

    pub fn index_to_action(index: usize) -> Option<Action> {
        if index < HAND_PLAY_OFFSET {
            return None;
        }
        if index < STOCK_PLAY_OFFSET {
            let relative = index - HAND_PLAY_OFFSET;
            let hand_index = relative / BUILD_PILE_COUNT;
            let build_pile = relative % BUILD_PILE_COUNT;
            return Some(Action::Play {
                source: CardSource::Hand(hand_index),
                build_pile,
            });
        }
        if index < DISCARD_PLAY_OFFSET {
            let build_pile = index - STOCK_PLAY_OFFSET;
            if build_pile < BUILD_PILE_COUNT {
                return Some(Action::Play {
                    source: CardSource::Stock,
                    build_pile,
                });
            }
            return None;
        }
        if index < DISCARD_OFFSET {
            let relative = index - DISCARD_PLAY_OFFSET;
            let discard_pile = relative / BUILD_PILE_COUNT;
            let build_pile = relative % BUILD_PILE_COUNT;
            return Some(Action::Play {
                source: CardSource::Discard(discard_pile),
                build_pile,
            });
        }
        if index < END_TURN_INDEX {
            let relative = index - DISCARD_OFFSET;
            let hand_index = relative / DISCARD_PILE_COUNT;
            let discard_pile = relative % DISCARD_PILE_COUNT;
            return Some(Action::Discard {
                hand_index,
                discard_pile,
            });
        }
        if index == END_TURN_INDEX {
            Some(Action::EndTurn)
        } else {
            None
        }
    }

    pub fn mask(legal: &[Action]) -> [f32; MAX_ACTIONS] {
        const NEGATIVE: f32 = -1.0e9;
        let mut mask = [NEGATIVE; MAX_ACTIONS];
        for action in legal {
            if let Some(index) = Self::action_index(action) {
                mask[index] = 0.0;
            }
        }
        mask
    }

    pub fn mask_tensor<B>(legal: &[Action]) -> Tensor<B, 2>
    where
        B: Backend,
        B::Device: Default,
    {
        let mask = Self::mask(legal);
        Tensor::<B, 2>::from_data(TensorData::from([mask]), &B::Device::default())
    }

    pub fn targets_from_indices(indices: &[usize]) -> [f32; MAX_ACTIONS] {
        let mut target = [0.0f32; MAX_ACTIONS];
        if indices.is_empty() {
            return target;
        }
        let weight = 1.0 / indices.len() as f32;
        for &idx in indices {
            if idx < MAX_ACTIONS {
                target[idx] = weight;
            }
        }
        target
    }

    pub fn target_tensor<B>(indices: &[usize]) -> Tensor<B, 2>
    where
        B: Backend,
        B::Device: Default,
    {
        let target = Self::targets_from_indices(indices);
        Tensor::<B, 2>::from_data(TensorData::from([target]), &B::Device::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;

    #[test]
    fn action_index_round_trip() {
        for index in 0..MAX_ACTIONS {
            if let Some(action) = ActionSpace::index_to_action(index) {
                let encoded = ActionSpace::action_index(&action).expect("encoded");
                assert_eq!(encoded, index);
            }
        }
    }

    #[test]
    fn mask_marks_only_legal_actions() {
        let actions = vec![
            Action::EndTurn,
            Action::Discard {
                hand_index: 1,
                discard_pile: 0,
            },
        ];
        let mask = ActionSpace::mask(&actions);
        let end_index = ActionSpace::action_index(&Action::EndTurn).unwrap();
        assert_eq!(mask[end_index], 0.0);
        let discard_index = ActionSpace::action_index(&actions[1]).unwrap();
        assert_eq!(mask[discard_index], 0.0);
        for (idx, value) in mask.iter().enumerate() {
            if idx != end_index && idx != discard_index {
                assert_eq!(*value, -1.0e9);
            }
        }
    }
}
