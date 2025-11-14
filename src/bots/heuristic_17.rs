use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::card::{Card, MAX_CARD_VALUE};
use crate::state::{GameStateView, PlayerPublicState};

/// Heuristic 17 bot ("build-pile chooser"): play priority from Heuristic 15 with
/// build-pile scoring so duplicate options pick the most promising pile.
pub struct Heuristic17Bot;

impl Heuristic17Bot {
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

    fn card_from_source(state: &GameStateView, source: CardSource) -> Option<Card> {
        match source {
            CardSource::Hand(index) => state.hand.get(index).copied(),
            CardSource::Stock => Self::self_player(state).stock_top,
            CardSource::Discard(index) => Self::self_player(state)
                .discard_piles
                .get(index)
                .and_then(|pile| pile.last())
                .copied(),
        }
    }

    /// Simple positional scoring: prefer piles with more progress and those about to recycle.
    fn score_play(state: &GameStateView, source: CardSource, build_pile: usize) -> i32 {
        let Some(pile) = state.build_piles.get(build_pile) else {
            return i32::MIN / 2;
        };
        let Some(card) = Self::card_from_source(state, source) else {
            return i32::MIN / 2;
        };
        let progress_bonus = (pile.cards.len() as i32) * 150;
        let closeness_bonus = (pile.next_value as i32) * 60;
        let completion_bonus = if pile.next_value == MAX_CARD_VALUE {
            500
        } else {
            0
        };
        let card_value = match card {
            Card::Number(value) => value as i32 * 30,
            Card::SkipBo => (MAX_CARD_VALUE as i32 + 2) * 30,
        };
        progress_bonus + closeness_bonus + completion_bonus + card_value - (build_pile as i32)
    }

    fn choose_best_play<F>(
        state: &GameStateView,
        legal_actions: &[Action],
        mut filter: F,
    ) -> Option<Action>
    where
        F: FnMut(&Action) -> bool,
    {
        let mut best_action: Option<&Action> = None;
        let mut best_score = i32::MIN;
        let mut best_build = usize::MAX;
        let mut best_hand = usize::MAX;
        for action in legal_actions {
            if !filter(action) {
                continue;
            }
            if let Action::Play { source, build_pile } = action {
                let score = Self::score_play(state, *source, *build_pile);
                let hand_index = match source {
                    CardSource::Hand(index) => *index,
                    _ => usize::MAX,
                };
                let is_better = if score > best_score {
                    true
                } else if score == best_score {
                    *build_pile < best_build
                        || (*build_pile == best_build && hand_index < best_hand)
                } else {
                    false
                };
                if is_better {
                    best_score = score;
                    best_build = *build_pile;
                    best_hand = hand_index;
                    best_action = Some(action);
                }
            }
        }
        best_action.cloned()
    }

    fn best_stock_play(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        Self::choose_best_play(state, legal_actions, |action| {
            matches!(
                action,
                Action::Play {
                    source: CardSource::Stock,
                    ..
                }
            )
        })
    }

    fn best_hand_play(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        Self::choose_best_play(state, legal_actions, |action| {
            matches!(
                action,
                Action::Play {
                    source: CardSource::Hand(_),
                    ..
                }
            )
        })
    }

    fn best_discard_play(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        Self::choose_best_play(state, legal_actions, |action| {
            matches!(
                action,
                Action::Play {
                    source: CardSource::Discard(_),
                    ..
                }
            )
        })
    }
}

impl Default for Heuristic17Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic17Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 17 bot requires at least one legal action"
        );
        if let Some(action) = Self::best_stock_play(state, legal_actions) {
            return action;
        }
        if let Some(action) = Self::best_hand_play(state, legal_actions) {
            return action;
        }
        if let Some(action) = Self::best_discard_play(state, legal_actions) {
            return action;
        }
        if let Some(action) = legal_actions.iter().find(|a| matches!(a, Action::EndTurn)) {
            return action.clone();
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
