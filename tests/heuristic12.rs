use skipbot::Bot;
use skipbot::action::Action;
use skipbot::bots::heuristic_12::Heuristic12Bot;
use skipbot::card::Card;
use skipbot::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};

#[test]
fn heuristic12_prefers_one_below_discard() {
    // Build minimal game state snapshot with only discard actions legal.
    let settings = GameSettings::new(2).unwrap();
    let build_piles = [
        BuildPileView::empty(),
        BuildPileView::empty(),
        BuildPileView::empty(),
        BuildPileView::empty(),
    ];

    // Self player discard tops: pile 0 has a 5 (so discarding 4 gets one-below bonus),
    // pile 1 has a 10 (irrelevant), others empty.
    let self_player = PlayerPublicState {
        id: 0,
        stock_count: 30,
        stock_top: None,
        discard_tops: [Some(Card::Number(5)), Some(Card::Number(10)), None, None],
        discard_counts: [1, 1, 0, 0],
        hand_size: 2,
        is_current: true,
        has_won: false,
    };
    let other_player = PlayerPublicState {
        id: 1,
        stock_count: 30,
        stock_top: None,
        discard_tops: [None, None, None, None],
        discard_counts: [0, 0, 0, 0],
        hand_size: 0,
        is_current: false,
        has_won: false,
    };

    let state = GameStateView {
        settings,
        phase: TurnPhase::AwaitingAction,
        status: GameStatus::Ongoing,
        self_player: 0,
        current_player: 0,
        draw_pile_count: 0,
        recycle_pile_count: 0,
        build_piles,
        players: vec![self_player, other_player],
        hand: vec![Card::Number(4), Card::Number(9)],
    };

    // Legal discard actions: choose where to place each hand card.
    let legal_actions: Vec<Action> = vec![
        Action::Discard {
            hand_index: 0,
            discard_pile: 0,
        }, // one-below bonus (4 under 5)
        Action::Discard {
            hand_index: 0,
            discard_pile: 2,
        }, // no bonus
        Action::Discard {
            hand_index: 1,
            discard_pile: 1,
        }, // no bonus
    ];

    let mut bot = Heuristic12Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Discard {
            hand_index: 0,
            discard_pile: 0
        },
        "Bot should prefer discarding the one-below card onto pile with top 5"
    );
}
