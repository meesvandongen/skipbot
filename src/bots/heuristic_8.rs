use crate::action::{Action, CardSource};
use crate::bot::Bot;
use crate::card::{Card, MAX_CARD_VALUE, MIN_CARD_VALUE};
use crate::state::{GameStateView, PlayerPublicState};

/// Heuristic 8 bot (based on Heuristic 5)
/// Difference from Heuristic 5:
/// - It will not play a card to a build pile if doing so would make the next
///   player's stock card immediately playable (i.e., the next value after the
///   one we would play equals the next player's numeric stock value).
pub struct Heuristic8Bot;

impl Heuristic8Bot {
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
    /// that matches the next player's numeric stock-top card; such plays should
    /// be avoided by this heuristic.
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
            CardSource::Discard(d) => match Self::self_player(state).discard_tops[d] {
                Some(Card::Number(v)) => v,
                Some(Card::SkipBo) => pile.next_value,
                None => return false,
            },
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
        let existing_top = player.discard_tops.get(discard_pile).and_then(|v| *v);
        let duplicate_bonus = if existing_top == Some(card) { 600 } else { 0 };
        let pile_depth = player
            .discard_counts
            .get(discard_pile)
            .copied()
            .unwrap_or_default() as i32;
        let spacing_penalty = pile_depth * 20;
        1_000 + duplicate_bonus - spacing_penalty - (hand_index as i32 * 10)
    }

    /// Compute the sequence of required values that must be played (in order)
    /// on a specific build pile so that the stock card becomes playable on that pile.
    /// Returns empty list when stock can be played immediately.
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

    /// Try to plan a minimal sequence of plays (using hand + discard tops) to make the stock
    /// card playable, and return the first action of that plan if it's feasible.
    fn can_play_stock(state: &GameStateView, legal_actions: &[Action]) -> Option<Action> {
        let player = Self::self_player(state);
        let stock = player.stock_top?;

        // Fast path: if stock is immediately playable on any pile, play it.
        if let Card::SkipBo = stock {
            // Choose a pile with the highest next_value (closest to completion).
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
                if legal_actions.contains(&action) && !Self::should_block_play(state, &action) {
                    return Some(action);
                }
            }
        } else if let Card::Number(s) = stock {
            // If any pile already needs s, we can play stock now.
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
                if legal_actions.contains(&action) && !Self::should_block_play(state, &action) {
                    return Some(action);
                }
            }
        }

        // Build counts and index maps for available cards: hand and discard tops.
        #[derive(Clone, Copy)]
        enum SourceKind {
            Hand(usize),
            Discard(usize),
        }

        // For quick lookup by required value, store lists of sources that can satisfy that value.
        // We'll prefer exact-number matches from discards, then hand, then Skip-Bo (discard first).
        let mut by_value: [Vec<SourceKind>; (MAX_CARD_VALUE as usize) + 1] = Default::default();
        // Index 0 unused; values 1..=12 used.

        // Track skip-bo sources separately for prioritization.
        let mut skipbo_discards: Vec<usize> = Vec::new();
        let mut skipbo_hands: Vec<usize> = Vec::new();

        for (idx, card) in state.hand.iter().copied().enumerate() {
            match card {
                Card::Number(v) => by_value[v as usize].push(SourceKind::Hand(idx)),
                Card::SkipBo => skipbo_hands.push(idx),
            }
        }
        for (d_idx, top) in Self::self_player(state)
            .discard_tops
            .iter()
            .copied()
            .enumerate()
        {
            if let Some(card) = top {
                match card {
                    Card::Number(v) => by_value[v as usize].push(SourceKind::Discard(d_idx)),
                    Card::SkipBo => skipbo_discards.push(d_idx),
                }
            }
        }

        // Helper to attempt planning for a specific pile. Returns planned actions
        // (excluding the final stock play), or None if impossible.
        let mut best_plan: Option<(usize, Vec<Action>)> = None;

        for (pile_idx, pile) in state.build_piles.iter().enumerate() {
            let required = Self::required_values_for_pile(pile.next_value, stock);

            // Prepare working copies of availability for this pile's planning attempt.
            let mut used_hand: Vec<bool> = vec![false; state.hand.len()];
            let mut used_discard: [bool; 4] = [false; 4];
            let skipbo_discards_left = skipbo_discards.clone();
            let skipbo_hands_left = skipbo_hands.clone();

            let mut actions: Vec<Action> = Vec::new();
            let mut possible = true;

            // Current next_value progresses as we plan.
            let mut _sim_next = pile.next_value;

            for need in required.iter().copied() {
                // Try exact from discard first among those not used yet.
                let mut picked: Option<SourceKind> = None;
                for &src in &by_value[need as usize] {
                    match src {
                        SourceKind::Discard(d) if !used_discard[d] => {
                            picked = Some(src);
                            break;
                        }
                        _ => {}
                    }
                }
                // Then try exact from hand.
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
                // Then use a Skip-Bo from discard, else hand.
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

                // Record action and mark usage.
                match src {
                    SourceKind::Discard(d) => {
                        used_discard[d] = true;
                        let action = Action::Play {
                            source: CardSource::Discard(d),
                            build_pile: pile_idx,
                        };
                        if Self::should_block_play(state, &action) {
                            possible = false;
                            break;
                        }
                        actions.push(action);
                    }
                    SourceKind::Hand(h) => {
                        used_hand[h] = true;
                        let action = Action::Play {
                            source: CardSource::Hand(h),
                            build_pile: pile_idx,
                        };
                        if Self::should_block_play(state, &action) {
                            possible = false;
                            break;
                        }
                        actions.push(action);
                    }
                }

                // Advance simulated next value.
                _sim_next = if _sim_next == MAX_CARD_VALUE {
                    1
                } else {
                    _sim_next + 1
                };
            }

            if !possible {
                continue;
            }

            // Choose best plan by fewest required actions; tie-breaker: more progressed piles.
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

        // If we have a plan, return the first action (or the stock play if no prerequisites).
        if let Some((pile_idx, actions)) = best_plan {
            if actions.is_empty() {
                let action = Action::Play {
                    source: CardSource::Stock,
                    build_pile: pile_idx,
                };
                if legal_actions.contains(&action) && !Self::should_block_play(state, &action) {
                    return Some(action);
                }
            } else {
                let first = actions[0].clone();
                if legal_actions.contains(&first) && !Self::should_block_play(state, &first) {
                    return Some(first);
                }
            }
        }

        None
    }

    /// Extract the numeric card tied to a legal play action, if any.
    /// Skip-Bo cards are ignored for the numeric threshold rule (return None).
    fn played_card_value(state: &GameStateView, action: &Action) -> Option<u8> {
        if let Action::Play { source, .. } = action {
            match *source {
                CardSource::Hand(i) => match state.hand.get(i).copied()? {
                    Card::Number(v) => Some(v),
                    Card::SkipBo => None,
                },
                CardSource::Discard(d) => match Self::self_player(state).discard_tops[d]? {
                    Card::Number(v) => Some(v),
                    Card::SkipBo => None,
                },
                CardSource::Stock => match Self::self_player(state).stock_top? {
                    Card::Number(v) => Some(v),
                    Card::SkipBo => None,
                },
            }
        } else {
            None
        }
    }
}

impl Default for Heuristic8Bot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for Heuristic8Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic 8 bot requires at least one legal action"
        );

        // 1) Try stock-first plan.
        if let Some(action) = Self::can_play_stock(state, legal_actions) {
            return action;
        }

        // 2) Otherwise, play only number cards where value > stock value AND value >= 6.
        let stock_value = match Self::self_player(state).stock_top {
            Some(Card::Number(v)) => v,
            Some(Card::SkipBo) | None => 1, // treat Skip-Bo/None as minimal numeric baseline
        };

        let mut best_play: Option<(u8, usize, Action)> = None; // (card_value, pile_len, action)
        for action in legal_actions.iter() {
            if !matches!(action, Action::Play { .. }) {
                continue;
            }
            if Self::should_block_play(state, action) {
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

        // 3) If no qualifying plays, choose a discard as in heuristic 3.
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

        // 4) Fallback: if nothing else, pick the first legal non-blocked action if any.
        if let Some(action) = legal_actions
            .iter()
            .find(|a| !matches!(a, Action::Play { .. }) || !Self::should_block_play(state, a))
        {
            return action.clone();
        }

        // If all remaining are blocked plays, we must pick something; choose the first legal action.
        legal_actions[0].clone()
    }
}
