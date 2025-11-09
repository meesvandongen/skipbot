//! Scoring utilities for Skip-Bo simulations.
//!
//! Current scoring rule (winner-only):
//!   points = 25 (base win) + 5 * (sum of opponents' remaining stock cards)
//! Non-winning players receive 0 points.
//! Drawn / aborted games award no points.

use crate::action::PlayerId;
use crate::state::GameStateView;

/// Compute winner's points for a finished game view.
///
/// Assumes `winner` is a valid player id present in `state.players`.
/// If the game was not won (draw/aborted), caller should skip calling this.
pub fn winner_points(state: &GameStateView, winner: PlayerId) -> usize {
    let mut opponents_stock_total = 0usize;
    for p in &state.players {
        if p.id != winner {
            opponents_stock_total += p.stock_count;
        }
    }
    25 + 5 * opponents_stock_total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{BuildPileView, GameSettings, GameStatus, PlayerPublicState, TurnPhase};

    fn dummy_state(stock_counts: &[usize], winner: PlayerId) -> GameStateView {
        let players: Vec<PlayerPublicState> = stock_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| PlayerPublicState {
                id: i,
                stock_count: c,
                stock_top: None,
                discard_piles: [vec![], vec![], vec![], vec![]],
                hand_size: 0,
                is_current: false,
                has_won: i == winner,
            })
            .collect();
        GameStateView {
            settings: GameSettings {
                num_players: stock_counts.len(),
                stock_size: 30,
                hand_size: 5,
                discard_piles: 4,
                build_piles: 4,
            },
            phase: TurnPhase::GameOver,
            status: GameStatus::Finished { winner },
            self_player: winner,
            current_player: winner,
            draw_pile_count: 0,
            recycle_pile_count: 0,
            build_piles: [
                BuildPileView::empty(),
                BuildPileView::empty(),
                BuildPileView::empty(),
                BuildPileView::empty(),
            ],
            players,
            hand: Vec::new(),
        }
    }

    #[test]
    fn test_winner_points_three_players() {
        // Winner index 1, opponents have 10 and 3 stock cards => 25 + 5*(13) = 90
        let state = dummy_state(&[10, 0, 3], 1);
        assert_eq!(winner_points(&state, 1), 90);
    }

    #[test]
    fn test_winner_points_two_players() {
        // Winner index 0, opponent has 7 => 25 + 5*(7) = 60
        let state = dummy_state(&[0, 7], 0);
        assert_eq!(winner_points(&state, 0), 60);
    }

    #[test]
    fn test_winner_points_all_opponents_empty() {
        // Winner index 2, opponents have 0 stock => base 25
        let state = dummy_state(&[0, 0, 0, 0], 2);
        assert_eq!(winner_points(&state, 2), 25);
    }
}
