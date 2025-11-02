
/// Rule-based bot that plays "sensible" moves without search or learning.
///
/// In plain English:
/// - Try to play from the stock pile whenever possible (finishing your stock wins the game),
///   then from discard piles, then from the hand.
/// - Prefer plays that push build piles forward, especially when they are already long or
///   close to completion.
/// - Treat Skip-Bo as a wild card with slightly higher priority than any numbered card.
/// - When discarding, prefer stacking the same number on the same discard pile to set up future
///   multi-plays, and avoid making discard piles too deep.
/// - Ending the turn is strongly discouraged if there are any beneficial plays.
pub struct HeuristicBot;

impl HeuristicBot {
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

    /// Score how good it is to play a given card (from hand/stock/discard) onto a specific build pile.
    /// Larger scores are better. The final heuristic combines several intuitive pieces:
    /// - Source bonus: prefer playing from stock > discard > hand.
    /// - Card value: higher numbers are slightly preferred; Skip-Bo gets a small bonus as a wild.
    /// - Pile progress: longer build piles and piles closer to the target value are preferred.
    /// - Completion nudge: small bump when the pile is about to complete.
    fn score_play(state: &GameStateView, source: CardSource, build_pile: usize) -> i32 {
        let Some(pile) = state.build_piles.get(build_pile) else {
            return i32::MIN / 2;
        };
        let card = match source {
            CardSource::Hand(index) => state.hand.get(index).copied(),
            CardSource::Stock => Self::self_player(state).stock_top,
            CardSource::Discard(index) => {
                let player = Self::self_player(state);
                player.discard_tops.get(index).copied().flatten()
            }
        };
        let Some(card) = card else {
            return i32::MIN / 2;
        };
        // Prefer freeing critical sources: stock most, then discard, then hand.
        let source_bonus = match source {
            CardSource::Stock => 10_000,
            CardSource::Discard(_) => 4_000,
            CardSource::Hand(_) => 2_000,
        };
        // Slightly prefer higher-value cards; Skip-Bo counts as the highest.
        let value_score = Self::card_priority(card) * 60;
        // Encourage advancing piles that are already underway.
        let progress_bonus = (pile.cards.len() as i32) * 40;
        // Favor moves that bring the pile closer to completion.
        let closeness_bonus = (pile.next_value as i32) * 25;
        // Small nudge when a pile is one step from wrapping/completing.
        let completion_bonus = if pile.next_value == MAX_CARD_VALUE {
            1_000
        } else {
            0
        };
        // Skip-Bo gets a modest extra reward as a flexible wildcard.
        let wild_bonus = if matches!(card, Card::SkipBo) { 300 } else { 0 };
        source_bonus
            + value_score
            + progress_bonus
            + closeness_bonus
            + completion_bonus
            + wild_bonus
    }

    /// Score how good it is to discard a specific hand card onto a chosen discard pile.
    /// Heuristics used:
    /// - Stack duplicates on the same pile to set up future chains.
    /// - Prefer keeping discard piles shallow (penalize deep piles).
    /// - Slightly prefer higher-priority cards (Skip-Bo > higher numbers > lower numbers) as discards
    ///   when no good plays exist, but the depth penalty dominates.
    /// - Tiny tie-breaker toward lower hand indices for stability/readability.
    fn score_discard(state: &GameStateView, hand_index: usize, discard_pile: usize) -> i32 {
        let Some(card) = state.hand.get(hand_index).copied() else {
            return i32::MIN / 2;
        };
        let player = Self::self_player(state);
        let existing_top = player
            .discard_tops
            .get(discard_pile)
            .and_then(|value| *value);
        // Reward stacking the same value to enable future multi-plays.
        let duplicate_bonus = if existing_top == Some(card) { 600 } else { 0 };
        let pile_depth = player
            .discard_counts
            .get(discard_pile)
            .copied()
            .unwrap_or_default() as i32;
        // Slight preference by intrinsic card priority; dominated by depth control.
        let priority = Self::card_priority(card) * 12;
        // Discourage making discard stacks too tall/hard to free later.
        let spacing_penalty = pile_depth * 20;
        // Small base to make any discard acceptable when nothing better is available.
        // Minor tie-break by hand slot index to keep behavior stable.
        1_000 + duplicate_bonus + priority - spacing_penalty - (hand_index as i32 * 10)
    }

    /// Map a legal action to a score: prefer plays and good discards; avoid ending turn early.
    fn score_action(state: &GameStateView, action: &Action) -> i32 {
        match action {
            Action::Play { source, build_pile } => Self::score_play(state, *source, *build_pile),
            Action::Discard {
                hand_index,
                discard_pile,
            } => Self::score_discard(state, *hand_index, *discard_pile),
            // Strong penalty: if any useful move exists, don't end the turn yet.
            Action::EndTurn => -5_000,
        }
    }
}

impl Default for HeuristicBot {
    fn default() -> Self {
        Self::new()
    }
}

impl Bot for HeuristicBot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "heuristic bot requires at least one legal action"
        );
        legal_actions
            .iter()
            .max_by_key(|action| Self::score_action(state, action))
            .cloned()
            .unwrap_or_else(|| legal_actions[0].clone())
    }
}