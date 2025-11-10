use skipbot::Bot;
use skipbot::action::{Action, CardSource};
use skipbot::bots::heuristic_14::Heuristic14Bot;
use skipbot::card::Card;
use skipbot::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};

// Helper to build a minimal state for tests.
fn base_state(hand: Vec<Card>, discard_tops: [Vec<Card>; 4], stock_top: Option<Card>) -> GameStateView {
    let settings = GameSettings::new(2).unwrap();
    let build_piles = [
        BuildPileView::empty(),
        BuildPileView { cards: vec![Card::Number(1)], next_value: 2 },
        BuildPileView::empty(),
        BuildPileView::empty(),
    ];
    let self_player = PlayerPublicState {
        id: 0,
        stock_count: 30,
        stock_top,
        discard_piles: discard_tops,
        hand_size: hand.len(),
        is_current: true,
        has_won: false,
    };
    let other_player = PlayerPublicState {
        id: 1,
        stock_count: 30,
        stock_top: None,
        discard_piles: [vec![], vec![], vec![], vec![]],
        hand_size: 0,
        is_current: false,
        has_won: false,
    };
    GameStateView {
        settings,
        phase: TurnPhase::AwaitingAction,
        status: GameStatus::Ongoing,
        self_player: 0,
        current_player: 0,
        draw_pile_count: 0,
        recycle_pile_count: 0,
        build_piles,
        players: vec![self_player, other_player],
        hand,
    }
}

#[test]
fn heuristic14_prefers_stock_play() {
    // Stock is playable (value 2 on pile with next_value 2)
    let state = base_state(vec![Card::Number(9)], [vec![], vec![], vec![], vec![]], Some(Card::Number(2)));
    let legal_actions = vec![
        Action::Play { source: CardSource::Stock, build_pile: 1 },
        Action::Play { source: CardSource::Hand(0), build_pile: 1 },
        Action::Discard { hand_index: 0, discard_pile: 0 },
    ];
    let mut bot = Heuristic14Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::Play { source: CardSource::Stock, build_pile: 1 });
}

#[test]
fn heuristic14_prefers_discard_play_over_hand() {
    // Discard top playable (2), hand has playable (2) also.
    let state = base_state(
        vec![Card::Number(2), Card::Number(7)],
        [vec![Card::Number(2)], vec![], vec![], vec![]],
        None,
    );
    let legal_actions = vec![
        Action::Play { source: CardSource::Discard(0), build_pile: 1 },
        Action::Play { source: CardSource::Hand(0), build_pile: 1 },
        Action::Discard { hand_index: 1, discard_pile: 0 },
    ];
    let mut bot = Heuristic14Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::Play { source: CardSource::Discard(0), build_pile: 1 });
}

#[test]
fn heuristic14_plays_hand_when_only_option() {
    let state = base_state(
        vec![Card::Number(2)],
        [vec![], vec![], vec![], vec![]],
        None,
    );
    let legal_actions = vec![
        Action::Play { source: CardSource::Hand(0), build_pile: 1 },
        Action::Discard { hand_index: 0, discard_pile: 0 },
    ];
    let mut bot = Heuristic14Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::Play { source: CardSource::Hand(0), build_pile: 1 });
}

#[test]
fn heuristic14_ends_turn_when_no_plays() {
    // Empty hand, only EndTurn is relevant.
    let state = base_state(vec![], [vec![], vec![], vec![], vec![]], None);
    let legal_actions = vec![Action::EndTurn];
    let mut bot = Heuristic14Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::EndTurn);
}

#[test]
fn heuristic14_discards_when_no_play_and_hand_not_empty() {
    // Hand has no playable (needs 2, we only have 3). Must discard.
    let state = base_state(
        vec![Card::Number(3)],
        [vec![], vec![], vec![], vec![]],
        None,
    );
    let legal_actions = vec![
        Action::Discard { hand_index: 0, discard_pile: 2 },
    ];
    let mut bot = Heuristic14Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::Discard { hand_index: 0, discard_pile: 2 });
}
