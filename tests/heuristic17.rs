use skipbot::action::{Action, CardSource};
use skipbot::bot::Bot;
use skipbot::bots::heuristic_17::Heuristic17Bot;
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
fn heuristic17_ranks_stock_targets() {
    let mut pile_one = BuildPileView::empty();
    pile_one
        .cards
        .extend_from_slice(&[Card::Number(1), Card::Number(2), Card::Number(3)]);
    pile_one.next_value = 4;
    let state = base_state(
        vec![],
        [vec![], vec![], vec![], vec![]],
        Some(Card::SkipBo),
        [
            BuildPileView::empty(),
            pile_one,
            BuildPileView::empty(),
            BuildPileView::empty(),
        ],
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Stock,
            build_pile: 0,
        },
        Action::Play {
            source: CardSource::Stock,
            build_pile: 1,
        },
    ];
    let mut bot = Heuristic17Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Play {
            source: CardSource::Stock,
            build_pile: 1
        }
    );
}

#[test]
fn heuristic17_prefers_better_hand_target() {
    let mut pile_zero = BuildPileView::empty();
    pile_zero.cards.push(Card::Number(1));
    pile_zero.next_value = 2;
    let state = base_state(
        vec![Card::Number(2), Card::Number(2)],
        [vec![], vec![], vec![], vec![]],
        None,
        [
            pile_zero.clone(),
            BuildPileView::empty(),
            BuildPileView::empty(),
            BuildPileView::empty(),
        ],
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0,
        },
        Action::Play {
            source: CardSource::Hand(1),
            build_pile: 1,
        },
    ];
    let mut bot = Heuristic17Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0
        }
    );
}

#[test]
fn heuristic17_keeps_hand_over_discard_priority() {
    let state = base_state(
        vec![Card::Number(3)],
        [vec![Card::Number(3)], vec![], vec![], vec![]],
        None,
        [
            BuildPileView::empty(),
            BuildPileView::empty(),
            BuildPileView::empty(),
            BuildPileView::empty(),
        ],
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Discard(0),
            build_pile: 2,
        },
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 2,
        },
    ];
    let mut bot = Heuristic17Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 2
        }
    );
}
