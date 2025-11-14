use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::card::{Card, MAX_CARD_VALUE};
use crate::state::{GameStateView, PlayerPublicState};

/// Heuristic 16 bot ("discard manager"): baseline play priorities with improved discard selection.
/// The bot keeps the naive play ordering from Heuristic 14 but scores discard destinations so the
/// hand stays organized for future turns.
pub struct Heuristic16Bot;

impl Heuristic16Bot {
    pub fn new() -> Self {
        Self
    }

    fn self_player<'a>(state: &'a GameStateView) -> &'a PlayerPublicState {
        state
            .players
            .iter()
            .find(|player| player.id == state.self_player)
            .expect("self player state must be present")
    }

    fn card_priority(card: Card) -> i32 {
        match card {
            Card::Number(value) => value as i32,
            Card::SkipBo => (MAX_CARD_VALUE as i32) + 1,
        }
    }

    /// Score discards by stacking duplicates, keeping piles shallow, and being stable under ties.
    fn score_discard(
        state: &GameStateView,
        player: &PlayerPublicState,
        hand_index: usize,
        discard_pile: usize,
    ) -> i32 {
        let Some(card) = state.hand.get(hand_index).copied() else {
            return i32::MIN / 2;
        };
        let Some(pile) = player.discard_piles.get(discard_pile) else {
            return i32::MIN / 2;
        };
        let top = pile.last().copied();
        let duplicate_bonus = if top == Some(card) { 700 } else { 0 };
        let depth_penalty = (pile.len() as i32) * 40;
        let priority = Self::card_priority(card) * 15;
        let empty_bonus = if pile.is_empty() { 80 } else { 0 };
        5_000 + duplicate_bonus + empty_bonus + priority
            - depth_penalty
            - (hand_index as i32 * 5)
            - (discard_pile as i32)
    }

    fn best_discard_action(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        let player = Self::self_player(state);
        let mut best_action: Option<&Action> = None;
        let mut best_score = i32::MIN;
        let mut best_pile = usize::MAX;
        let mut best_hand_index = usize::MAX;
        for action in legal_actions {
            if let Action::Discard {
                hand_index,
                discard_pile,
            } = action
            {
                let score = Self::score_discard(state, player, *hand_index, *discard_pile);
                let is_better = if score > best_score {
                    true
                } else if score == best_score {
                    *discard_pile < best_pile
                        || (*discard_pile == best_pile && *hand_index < best_hand_index)
                } else {
                    false
                };
                if is_better {
                    best_score = score;
                    best_pile = *discard_pile;
                    best_hand_index = *hand_index;
                    best_action = Some(action);
                }
            }
        }
        best_action.cloned()
    }
}

impl Default for Heuristic16Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic16Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 16 bot requires at least one legal action"
        );
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Stock,
                    ..
                }
            )
        }) {
            return action.clone();
        }
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Discard(_),
                    ..
                }
            )
        }) {
            return action.clone();
        }
        if let Some(action) = legal_actions.iter().find(|a| {
            matches!(
                a,
                Action::Play {
                    source: CardSource::Hand(_),
                    ..
                }
            )
        }) {
            return action.clone();
        }
        if let Some(action) = legal_actions.iter().find(|a| matches!(a, Action::EndTurn)) {
            return action.clone();
        }
        if let Some(discard_action) = Self::best_discard_action(state, legal_actions) {
            return discard_action;
        }
        if let Some(action) = legal_actions
            .iter()
            .find(|a| matches!(a, Action::Discard { .. }))
        {
            return action.clone();
        }
        legal_actions[0].clone()
    }
}
