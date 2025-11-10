use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::card::{Card, MAX_CARD_VALUE, MIN_CARD_VALUE};
use crate::state::{GameStateView, PlayerPublicState};

/// Heuristic 10 bot (based on Heuristic 9)
/// Differences from Heuristic 9:
/// - Stock-first planning phase is identical (no blocking there).
/// - In phase 2 (playing remaining cards) we avoid plays that would "break"
///   a duplicated next-value across build piles (i.e., if there are exactly two
///   piles with the same next_value, we avoid advancing one of them). Example:
///   - Two empty piles (next_value == 1): avoid playing a 1 on either.
///   - Two piles both at next_value == 7: avoid playing a 7 to turn one into 8.
///   This preserves duplicated next_values to reduce options for the next player.
pub struct Heuristic10Bot;

impl Heuristic10Bot {
    pub fn new() -> Self {
        Self
    }

    fn self_player<'a>(state: &'a GameStateView) -> &'a PlayerPublicState {
        state
            .players
            .iter()
            .find(|p| p.id == state.self_player)
            .expect("self player state must be present")
    }

    fn next_player<'a>(state: &'a GameStateView) -> Option<&'a PlayerPublicState> {
        let next_id = (state.current_player + 1) % state.settings.num_players;
        state.players.iter().find(|p| p.id == next_id)
    }

    /// Returns true if the given play action would set the pile to a next_value
    /// that matches the next player's numeric stock-top card.
    /// (Same as Heuristic 9; only used in fallback stage.)
    fn should_block_play(state: &GameStateView, action: &Action) -> bool {
        let Action::Play { source, build_pile } = action else {
            return false;
        };
        let pile = &state.build_piles[*build_pile];

        // Determine the numeric value we are effectively playing.
        let played_value = match *source {
            CardSource::Hand(i) => match state.hand.get(i).copied() {
                Some(Card::Number(v)) => v,
                Some(Card::SkipBo) => pile.next_value,
                None => return false,
            },
            CardSource::Discard(d) => {
                match Self::self_player(state).discard_piles[d].last().copied() {
                    Some(Card::Number(v)) => v,
                    Some(Card::SkipBo) => pile.next_value,
                    None => return false,
                }
            }
            CardSource::Stock => match Self::self_player(state).stock_top {
                Some(Card::Number(v)) => v,
                Some(Card::SkipBo) => pile.next_value,
                None => return false,
            },
        };

        let above = if played_value == MAX_CARD_VALUE {
            1
        } else {
            played_value + 1
        };
        if let Some(next_player) = Self::next_player(state) {
            matches!(next_player.stock_top, Some(Card::Number(v)) if v == above)
        } else {
            false
        }
    }

    /// Discard scoring: identical to heuristic_3 except we IGNORE card priority.
    fn score_discard(state: &GameStateView, hand_index: usize, discard_pile: usize) -> i32 {
        let Some(card) = state.hand.get(hand_index).copied() else {
            return i32::MIN / 2;
        };
        let player = Self::self_player(state);
        let existing_top = player
            .discard_piles
            .get(discard_pile)
            .and_then(|pile| pile.last())
            .copied();
        let duplicate_bonus = if existing_top == Some(card) { 600 } else { 0 };
        let pile_depth = player
            .discard_piles
            .get(discard_pile)
            .map(|p| p.len() as i32)
            .unwrap_or(0);
        let spacing_penalty = pile_depth * 20;
        1_000 + duplicate_bonus - spacing_penalty - (hand_index as i32 * 10)
    }

    /// Compute required sequential values for a pile until the stock becomes playable.
    fn required_values_for_pile(next_value: u8, stock: Card) -> Vec<u8> {
        match stock {
            Card::SkipBo => Vec::new(),
            Card::Number(s) => {
                if next_value == s {
                    return Vec::new();
                }
                let mut req = Vec::new();
                if next_value < s {
                    for v in next_value..s {
                        req.push(v);
                    }
                } else {
                    for v in next_value..=MAX_CARD_VALUE {
                        req.push(v);
                    }
                    for v in MIN_CARD_VALUE..s {
                        req.push(v);
                    }
                }
                req
            }
        }
    }

    /// Plan minimal prerequisite plays to make stock playable; return first action if feasible.
    /// (Identical logic to Heuristic 9; no blocking in this phase.)
    fn can_play_stock(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        let player = Self::self_player(state);
        let stock = player.stock_top?;

        // Fast path: immediate stock play.
        if let Card::SkipBo = stock {
            if let Some((best_idx, _)) = state
                .build_piles
                .iter()
                .enumerate()
                .max_by_key(|(_, pile)| (pile.next_value, pile.cards.len() as u8))
            {
                let action = Action::Play {
                    source: CardSource::Stock,
                    build_pile: best_idx,
                };
                if legal_actions.contains(&action) {
                    return Some(action);
                }
            }
        } else if let Card::Number(s) = stock {
            if let Some(best_idx) = state
                .build_piles
                .iter()
                .enumerate()
                .filter(|(_, pile)| pile.next_value == s)
                .map(|(i, pile)| (i, pile.cards.len()))
                .max_by_key(|(_, len)| *len)
                .map(|(i, _)| i)
            {
                let action = Action::Play {
                    source: CardSource::Stock,
                    build_pile: best_idx,
                };
                if legal_actions.contains(&action) {
                    return Some(action);
                }
            }
        }

        #[derive(Clone, Copy)]
        enum SourceKind {
            Hand(usize),
            Discard(usize),
        }

        let mut by_value: [Vec<SourceKind>; (MAX_CARD_VALUE as usize) + 1] = Default::default();
        let mut skipbo_discards: Vec<usize> = Vec::new();
        let mut skipbo_hands: Vec<usize> = Vec::new();

        for (idx, card) in state.hand.iter().copied().enumerate() {
            match card {
                Card::Number(v) => by_value[v as usize].push(SourceKind::Hand(idx)),
                Card::SkipBo => skipbo_hands.push(idx),
            }
        }
        for (d_idx, pile) in Self::self_player(state).discard_piles.iter().enumerate() {
            if let Some(card) = pile.last().copied() {
                match card {
                    Card::Number(v) => by_value[v as usize].push(SourceKind::Discard(d_idx)),
                    Card::SkipBo => skipbo_discards.push(d_idx),
                }
            }
        }

        let mut best_plan: Option<(usize, Vec<Action>)> = None;
        for (pile_idx, pile) in state.build_piles.iter().enumerate() {
            let required = Self::required_values_for_pile(pile.next_value, stock);
            let mut used_hand: Vec<bool> = vec![false; state.hand.len()];
            let mut used_discard: [bool; 4] = [false; 4];
            let skipbo_discards_left = skipbo_discards.clone();
            let skipbo_hands_left = skipbo_hands.clone();
            let mut actions: Vec<Action> = Vec::new();
            let mut possible = true;
            let mut _sim_next = pile.next_value;
            for need in required.iter().copied() {
                let mut picked: Option<SourceKind> = None;
                for &src in &by_value[need as usize] {
                    if let SourceKind::Discard(d) = src {
                        if !used_discard[d] {
                            picked = Some(src);
                            break;
                        }
                    }
                }
                if picked.is_none() {
                    for &src in &by_value[need as usize] {
                        if let SourceKind::Hand(h) = src {
                            if !used_hand[h] {
                                picked = Some(src);
                                break;
                            }
                        }
                    }
                }
                if picked.is_none() {
                    if let Some(d) = skipbo_discards_left
                        .iter()
                        .copied()
                        .find(|&d| !used_discard[d])
                    {
                        picked = Some(SourceKind::Discard(d));
                    } else if let Some(h) =
                        skipbo_hands_left.iter().copied().find(|&h| !used_hand[h])
                    {
                        picked = Some(SourceKind::Hand(h));
                    }
                }
                let Some(src) = picked else {
                    possible = false;
                    break;
                };
                match src {
                    SourceKind::Discard(d) => {
                        used_discard[d] = true;
                        actions.push(Action::Play {
                            source: CardSource::Discard(d),
                            build_pile: pile_idx,
                        });
                    }
                    SourceKind::Hand(h) => {
                        used_hand[h] = true;
                        actions.push(Action::Play {
                            source: CardSource::Hand(h),
                            build_pile: pile_idx,
                        });
                    }
                }
                _sim_next = if _sim_next == MAX_CARD_VALUE {
                    1
                } else {
                    _sim_next + 1
                };
            }
            if !possible {
                continue;
            }
            match &mut best_plan {
                None => best_plan = Some((pile_idx, actions)),
                Some((_best_idx, best_actions)) => {
                    if actions.len() < best_actions.len() {
                        *best_actions = actions;
                        *_best_idx = pile_idx;
                    }
                }
            }
        }
        if let Some((pile_idx, actions)) = best_plan {
            if actions.is_empty() {
                let action = Action::Play {
                    source: CardSource::Stock,
                    build_pile: pile_idx,
                };
                if legal_actions.contains(&action) {
                    return Some(action);
                }
            } else {
                let first = actions[0].clone();
                if legal_actions.contains(&first) {
                    return Some(first);
                }
            }
        }
        None
    }

    /// Extract numeric value played by a play action. Skip-Bo yields None.
    fn played_card_value(state: &GameStateView, action: &Action) -> Option<u8> {
        if let Action::Play { source, .. } = action {
            match *source {
                CardSource::Hand(i) => match state.hand.get(i).copied()? {
                    Card::Number(v) => Some(v),
                    Card::SkipBo => None,
                },
                CardSource::Discard(d) => {
                    match Self::self_player(state).discard_piles[d].last().copied()? {
                        Card::Number(v) => Some(v),
                        Card::SkipBo => None,
                    }
                }
                CardSource::Stock => match Self::self_player(state).stock_top? {
                    Card::Number(v) => Some(v),
                    Card::SkipBo => None,
                },
            }
        } else {
            None
        }
    }

    /// Returns true if a play action would break a duplicated next_value pair.
    /// Specifically: if the target pile's current next_value appears on exactly two
    /// piles (including this one), we consider advancing it as "breaking" duplication.
    fn breaks_pair_duplication(state: &GameStateView, action: &Action) -> bool {
        let Action::Play { build_pile, .. } = action else {
            return false;
        }; // non-play actions irrelevant
        let old_next = state.build_piles[*build_pile].next_value;
        let mut count = 0usize;
        for pile in state.build_piles.iter() {
            if pile.next_value == old_next {
                count += 1;
            }
        }
        count == 2
    }
}

impl Default for Heuristic10Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic10Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 10 bot requires at least one legal action"
        );

        // 1) Stock-first plan (same as heuristic 9).
        if let Some(action) = Self::can_play_stock(state, legal_actions) {
            return action;
        }

        // 2) Number play selection with duplication preservation.
        let stock_value = match Self::self_player(state).stock_top {
            Some(Card::Number(v)) => v,
            Some(Card::SkipBo) | None => 1,
        };
        let mut best_play: Option<(u8, usize, Action)> = None; // (card_value, pile_len, action)
        for action in legal_actions.iter() {
            if !matches!(action, Action::Play { .. }) {
                continue;
            }
            if Self::breaks_pair_duplication(state, action) {
                continue;
            }
            if let Some(v) = Self::played_card_value(state, action) {
                if v >= 6 && v > stock_value {
                    let pile_len = match action {
                        Action::Play { build_pile, .. } => {
                            state.build_piles[*build_pile].cards.len()
                        }
                        _ => 0,
                    };
                    let better = match &best_play {
                        None => true,
                        Some((best_v, best_len, _)) => {
                            v > *best_v || (v == *best_v && pile_len > *best_len)
                        }
                    };
                    if better {
                        best_play = Some((v, pile_len, action.clone()));
                    }
                }
            }
        }
        if let Some((_, _, action)) = best_play {
            return action;
        }

        // 3) Discard scoring.
        let mut best: Option<(i32, Action)> = None;
        for action in legal_actions.iter() {
            if let Action::Discard {
                hand_index,
                discard_pile,
            } = *action
            {
                let score = Self::score_discard(state, hand_index, discard_pile);
                if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                    best = Some((score, action.clone()));
                }
            }
        }
        if let Some((_, action)) = best {
            return action;
        }

        // 4) Fallback: prefer a non-blocked play or any non-play; also avoid blocked plays.
        if let Some(action) = legal_actions
            .iter()
            .find(|a| !matches!(a, Action::Play { .. }) || !Self::should_block_play(state, a))
        {
            return action.clone();
        }
        legal_actions[0].clone()
    }
}
