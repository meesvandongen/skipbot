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
    let game = GameBuilder::new(2)?
        .with_stock_size(5)
        .with_deck(deck)
        .build()?;
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

#[test]
fn refills_hand_after_emptying_during_play() -> Result<(), GameError> {
    // Prepare a draw sequence where the first five cards are 1..=5 so they can all be played.
    // Then provide five additional cards to be drawn immediately after the hand empties.
    let draw_sequence = vec![
        // These five will be used for the refill after emptying the hand
        Card::Number(12),
        Card::Number(12),
        Card::Number(12),
        Card::Number(12),
        Card::Number(12),
        // The last five (tail) will be drawn at the start of the turn via pop order,
        // resulting in a hand of [1,2,3,4,5] at indices 0..=4.
        Card::Number(5),
        Card::Number(4),
        Card::Number(3),
        Card::Number(2),
        Card::Number(1),
    ];
    // Minimal stock prefixes; the helper fills the rest with 12s as needed.
    let stock_p0 = vec![Card::Number(12)];
    let stock_p1 = vec![Card::Number(12)];
    let deck = build_deck(2, &draw_sequence, &[stock_p0, stock_p1]);

    let mut game = GameBuilder::new(2)?.with_deck(deck).build()?;
    let current = game.current_player();
    // Hand initially has 5 cards (1..=5) in ascending playable order due to pop order.
    for _ in 0..5 {
        game.apply_action(
            current,
            Action::Play {
                source: CardSource::Hand(0),
                build_pile: 0,
            },
        )?;
    }
    // After playing the 5th card from hand, the hand should be immediately refilled to 5 cards.
    let view = game.state_view(current)?;
    assert_eq!(view.build_piles[0].cards.len(), 5);
    assert_eq!(view.hand.len(), 5, "hand should be refilled to 5 after emptying during play");
    assert_eq!(view.draw_pile_count, 0, "refill should consume the remaining draw cards");
    Ok(())
}

#[test]
fn detects_stalemate_draw_when_no_draws_and_no_plays() -> Result<(), GameError> {
    // Construct a deck with no draw pile cards and stocks that cannot play (all 12s).
    // With empty hands and no draws, players can only EndTurn; after two rounds, it should be a draw.
    let num_players = 2;
    let draw_sequence: Vec<Card> = vec![];
    // Minimal stock prefixes; helper fills remaining with 12s, which are unplayable at game start (needs 1).
    let stock_p0 = vec![Card::Number(12)];
    let stock_p1 = vec![Card::Number(12)];
    let deck = build_deck(num_players, &draw_sequence, &[stock_p0, stock_p1]);
    let mut game = GameBuilder::new(num_players)?.with_deck(deck).build()?;

    // At start, player 0 has empty hand (no draws available), so only EndTurn is legal.
    for _ in 0..(num_players * 2) {
        let current = game.current_player();
        let actions = game.legal_actions(current)?;
        assert!(actions.iter().any(|a| matches!(a, skipbot::Action::EndTurn)));
        game.apply_action(current, skipbot::Action::EndTurn)?;
        if matches!(game.status(), GameStatus::Draw) {
            break;
        }
    }

    assert!(matches!(game.status(), GameStatus::Draw));
    assert!(game.is_finished());
    assert!(game.winner().is_none());
    Ok(())
}
