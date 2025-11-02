use skipbot::action::{Action, CardSource};
use skipbot::{Card, GameStatus};
use skipbot::{GameBuilder, GameError, card};

fn build_deck(
    num_players: usize,
    draw_sequence: &[Card],
    stock_prefixes: &[Vec<Card>],
) -> Vec<Card> {
    let settings = skipbot::GameSettings::new(num_players).expect("valid player count");
    assert_eq!(stock_prefixes.len(), num_players);
    let mut deck = Vec::new();
    deck.extend_from_slice(draw_sequence);
    for prefix in stock_prefixes.iter().rev() {
        let mut stock = prefix.clone();
        if stock.len() > settings.stock_size {
            panic!("stock prefix exceeds stock size");
        }
        stock.extend(std::iter::repeat(Card::Number(12)).take(settings.stock_size - stock.len()));
        deck.extend_from_slice(&stock);
    }
    deck
}

#[test]
fn initial_setup_two_players() -> Result<(), GameError> {
    let deck = card::full_deck();
    let deck_len = deck.len();
    let game = GameBuilder::new(2)?.with_deck(deck).build()?;
    let view0 = game.state_view(0)?;
    assert_eq!(view0.settings.num_players, 2);
    assert_eq!(view0.settings.stock_size, 30);
    assert_eq!(view0.hand.len(), 5);
    assert_eq!(view0.players[0].hand_size, 5);
    assert_eq!(view0.players[0].stock_count, 30);
    assert_eq!(view0.players[1].stock_count, 30);
    assert_eq!(view0.players[1].hand_size, 0);
    assert_eq!(view0.draw_pile_count, deck_len - (30 * 2) - 5);
    Ok(())
}

#[test]
fn stock_size_six_players() -> Result<(), GameError> {
    let deck = card::full_deck();
    let deck_len = deck.len();
    let game = GameBuilder::new(6)?.with_deck(deck).build()?;
    let view0 = game.state_view(0)?;
    assert_eq!(view0.settings.stock_size, 20);
    for player in &view0.players {
        assert_eq!(player.stock_count, 20);
    }
    assert_eq!(view0.hand.len(), 5);
    assert_eq!(view0.draw_pile_count, deck_len - (20 * 6) - 5);
    Ok(())
}

#[test]
fn configurable_stock_size_override() -> Result<(), GameError> {
    // Override to very small stock to simplify games.
    let deck = card::full_deck();
    let deck_len = deck.len();
    let game = GameBuilder::new(2)?.with_stock_size(5).with_deck(deck).build()?;
    let view0 = game.state_view(0)?;
    assert_eq!(view0.settings.stock_size, 5);
    assert_eq!(view0.players[0].stock_count, 5);
    assert_eq!(view0.players[1].stock_count, 5);
    assert_eq!(view0.hand.len(), 5);
    assert_eq!(view0.draw_pile_count, deck_len - (5 * 2) - 5);
    Ok(())
}

#[test]
fn build_pile_cycles_and_recycles() -> Result<(), GameError> {
    let draw_sequence = vec![
        Card::Number(5),
        Card::Number(4),
        Card::Number(3),
        Card::Number(2),
        Card::Number(1),
    ];
    let stock_p0 = vec![
        Card::Number(6),
        Card::Number(7),
        Card::Number(8),
        Card::Number(9),
        Card::Number(10),
        Card::Number(11),
        Card::Number(12),
        Card::Number(1),
    ];
    let stock_p1 = vec![Card::Number(12); 8];
    let deck = build_deck(2, &draw_sequence, &[stock_p0, stock_p1]);
    let mut game = GameBuilder::new(2)?.with_deck(deck).build()?;
    let current = game.current_player();
    assert_eq!(current, 0);
    for _ in 0..5 {
        game.apply_action(
            current,
            Action::Play {
                source: CardSource::Hand(0),
                build_pile: 0,
            },
        )?;
    }
    for _ in 0..7 {
        game.apply_action(
            current,
            Action::Play {
                source: CardSource::Stock,
                build_pile: 0,
            },
        )?;
    }
    let view = game.state_view(0)?;
    assert_eq!(view.build_piles[0].cards.len(), 0);
    assert_eq!(view.build_piles[0].next_value, 1);
    assert_eq!(view.recycle_pile_count, 12);
    assert_eq!(view.players[0].stock_count, view.settings.stock_size - 7);
    assert!(matches!(view.status, GameStatus::Ongoing));
    assert_eq!(view.hand.len(), 0);
    game.apply_action(current, Action::EndTurn)?;
    assert_eq!(game.current_player(), 1);
    let view1 = game.state_view(1)?;
    assert_eq!(view1.hand.len(), 5);
    assert_eq!(view1.draw_pile_count, 7);
    assert_eq!(view1.recycle_pile_count, 0);
    Ok(())
}

#[test]
fn skipbo_wildcard_support() -> Result<(), GameError> {
    let draw_sequence = vec![
        Card::Number(5),
        Card::Number(4),
        Card::Number(3),
        Card::SkipBo,
        Card::Number(1),
    ];
    let stock_p0 = vec![Card::Number(6)];
    let stock_p1 = vec![Card::Number(12)];
    let deck = build_deck(2, &draw_sequence, &[stock_p0, stock_p1]);
    let mut game = GameBuilder::new(2)?.with_deck(deck).build()?;
    let current = game.current_player();
    game.apply_action(
        current,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0,
        },
    )?; // plays 1
    game.apply_action(
        current,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0,
        },
    )?; // plays Skip-Bo as 2
    let view = game.state_view(0)?;
    assert_eq!(view.build_piles[0].cards.len(), 2);
    assert_eq!(view.build_piles[0].next_value, 3);
    game.apply_action(
        current,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0,
        },
    )?; // plays 3
    game.apply_action(
        current,
        Action::Play {
            source: CardSource::Hand(0),
            build_pile: 0,
        },
    )?; // plays 4
    let view = game.state_view(0)?;
    assert_eq!(view.build_piles[0].next_value, 5);
    Ok(())
}
