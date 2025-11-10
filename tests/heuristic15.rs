use skipbot::Bot;
use skipbot::action::{Action, CardSource};
use skipbot::bots::heuristic_15::Heuristic15Bot;
use skipbot::card::Card;
use skipbot::state::{
    BuildPileView, GameSettings, GameStateView, GameStatus, PlayerPublicState, TurnPhase,
};

fn base_state(
    hand: Vec<Card>,
    discard_tops: [Vec<Card>; 4],
    stock_top: Option<Card>,
) -> GameStateView {
    let settings = GameSettings::new(2).unwrap();
    let build_piles = [
        BuildPileView::empty(),
        BuildPileView {
            cards: vec![Card::Number(1)],
            next_value: 2,
        },
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
fn heuristic15_prefers_stock_play() {
    let state = base_state(
        vec![Card::Number(2)],
        [vec![], vec![], vec![], vec![]],
        Some(Card::Number(2)),
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Stock,
            build_pile: 1,
        },
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 1,
        },
        Action::Play {
            source: CardSource::Discard(0),
            build_pile: 1,
        },
    ];
    let mut bot = Heuristic15Bot::new();
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
fn heuristic15_prefers_hand_over_discard() {
    let state = base_state(
        vec![Card::Number(2)],
        [vec![Card::Number(2)], vec![], vec![], vec![]],
        None,
    );
    let legal_actions = vec![
        Action::Play {
            source: CardSource::Discard(0),
            build_pile: 1,
        },
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 1,
        },
    ];
    let mut bot = Heuristic15Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(
        chosen,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 1
        }
    );
}

#[test]
fn heuristic15_ends_turn_when_no_plays() {
    let state = base_state(vec![], [vec![], vec![], vec![], vec![]], None);
    let legal_actions = vec![Action::EndTurn];
    let mut bot = Heuristic15Bot::new();
    let chosen = bot.select_action(&state, &legal_actions);
    assert_eq!(chosen, Action::EndTurn);
}
