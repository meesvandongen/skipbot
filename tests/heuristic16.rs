use skipbot::action::{Action, CardSource};
use skipbot::bot::Bot;
use skipbot::bots::heuristic_16::Heuristic16Bot;
use skipbot::card::Card;
use skipbot::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};

fn base_state(
    hand: Vec<Card>,
    discard_piles: [Vec<Card>; 4],
    stock_top: Option<Card>,
    build_piles: [BuildPileView; 4],
) -> GameStateView {
    let settings = GameSettings::new(2).unwrap();
    let self_player = PlayerPublicState {
        id: 0,
        stock_count: 30,
        stock_top,
        discard_piles,
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
fn heuristic16_prefers_duplicate_discard() {
    let state = base_state(
        vec![Card::Number(4)],
        [vec![Card::Number(4)], vec![], vec![], vec![]],
        None,
        [
            BuildPileView::empty(),
            BuildPileView::empty(),
            BuildPileView::empty(),
            BuildPileView::empty(),
        ],
    );
    let legal_actions = vec![
        Action::Discard {
            hand_index: 0,
            discard_pile: 1,
        },
        Action::Discard {
            hand_index: 0,
            discard_pile: 0,
        },
    ];
    let mut bot = Heuristic16Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Discard {
            hand_index: 0,
            discard_pile: 0
        }
    );
}

#[test]
fn heuristic16_still_prioritizes_stock_play() {
    let mut build_two = BuildPileView::empty();
    build_two.cards.push(Card::Number(1));
    build_two.next_value = 2;
    let state = base_state(
        vec![Card::Number(5)],
        [vec![], vec![], vec![], vec![]],
        Some(Card::Number(2)),
        [
            BuildPileView::empty(),
            build_two,
            BuildPileView::empty(),
            BuildPileView::empty(),
        ],
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Stock,
            build_pile: 1,
        },
        Action::Discard {
            hand_index: 0,
            discard_pile: 0,
        },
    ];
    let mut bot = Heuristic16Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Play {
            source: CardSource::Stock,
            build_pile: 1
        }
    );
}
