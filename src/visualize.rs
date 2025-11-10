use std::fmt::Write;

use crate::action::{Action, CardSource};
use crate::card::Card;
use crate::state::{GameStateView, GameStatus};

/// Customize state rendering for CLI visualization.
#[derive(Clone, Copy, Debug)]
pub struct VisualOptions {
    pub show_build_sequences: bool,
    pub show_discard_sizes: bool,
}

impl Default for VisualOptions {
    fn default() -> Self {
        Self {
            show_build_sequences: true,
            show_discard_sizes: true,
        }
    }
}

/// Fine tune textual action descriptions.
#[derive(Clone, Copy, Debug)]
pub struct DescribeOptions {
    pub include_card_details: bool,
    pub include_build_expectation: bool,
}

impl Default for DescribeOptions {
    fn default() -> Self {
        Self {
            include_card_details: true,
            include_build_expectation: true,
        }
    }
}

pub fn render_state(state: &GameStateView) -> String {
    render_state_with_options(state, VisualOptions::default())
}

pub fn render_state_with_options(state: &GameStateView, options: VisualOptions) -> String {
    let mut out = String::new();
    let status = match state.status {
        GameStatus::Ongoing => String::from("Ongoing"),
        GameStatus::Finished { winner } => {
            format!("Finished (winner: Player {winner})")
        }
        GameStatus::Draw => String::from("Finished (draw)"),
    };
    let _ = writeln!(out, "Game status: {status}");
    let _ = writeln!(out, "Phase: {:?}", state.phase);
    let _ = writeln!(
        out,
        "Current player: {}{}",
        state.current_player,
        if state.current_player == state.self_player {
            " (You)"
        } else {
            ""
        }
    );
    let _ = writeln!(
        out,
        "Draw pile: {}  |  Recycle pile: {}",
        state.draw_pile_count, state.recycle_pile_count
    );
    let _ = writeln!(out, "Build piles:");
    for (idx, pile) in state.build_piles.iter().enumerate() {
        let sequence = if options.show_build_sequences && !pile.cards.is_empty() {
            let seq = pile
                .cards
                .iter()
                .map(|card| format_card(*card))
                .collect::<Vec<_>>()
                .join(" ");
            format!("[{seq}]")
        } else {
            String::from("[-]")
        };
        let _ = writeln!(out, "  [{idx}] next {}  {}", pile.next_value, sequence);
    }
    let _ = writeln!(out, "Players:");
    for player in &state.players {
        let label_you = if player.id == state.self_player {
            " (You)"
        } else {
            ""
        };
        let current_tag = if player.is_current { " <- current" } else { "" };
        let stock_top = player
            .stock_top
            .map(format_card)
            .unwrap_or_else(|| String::from("--"));
        let mut discard_parts = Vec::with_capacity(state.settings.discard_piles);
        for idx in 0..state.settings.discard_piles {
            let pile = &player.discard_piles[idx];
            let top = pile
                .last()
                .map(|c| format_card(*c))
                .unwrap_or_else(|| String::from("--"));
            if options.show_discard_sizes {
                discard_parts.push(format!("{}:{} ({})", idx, top, pile.len()));
            } else {
                discard_parts.push(format!("{}:{}", idx, top));
            }
        }
        let discard_display = discard_parts.join("  ");
        let _ = writeln!(
            out,
            "  Player {}{} - stock {} (top: {}){}",
            player.id, label_you, player.stock_count, stock_top, current_tag
        );
        let _ = writeln!(out, "    Discards: {discard_display}");
        if player.id == state.self_player {
            if state.hand.is_empty() {
                let _ = writeln!(out, "    Hand: (empty)");
            } else {
                let mut hand_entries = Vec::with_capacity(state.hand.len());
                for (idx, card) in state.hand.iter().enumerate() {
                    hand_entries.push(format!("{}:{}", idx, format_card(*card)));
                }
                let hand_display = hand_entries.join("  ");
                let _ = writeln!(out, "    Hand: {hand_display}");
            }
        } else {
            let _ = writeln!(out, "    Hand size: {}", player.hand_size);
        }
    }
    out
}

pub fn describe_action(state: &GameStateView, action: &Action) -> String {
    describe_action_with_options(state, action, DescribeOptions::default())
}

pub fn describe_action_with_options(
    state: &GameStateView,
    action: &Action,
    options: DescribeOptions,
) -> String {
    match action {
        Action::Play { source, build_pile } => {
            let pile_info = state
                .build_piles
                .get(*build_pile)
                .map(|pile| pile.next_value)
                .unwrap_or(0);
            let source_desc = match source {
                CardSource::Hand(index) => {
                    if let Some(card) = state.hand.get(*index) {
                        if options.include_card_details {
                            format!("hand[{index}] {}", format_card(*card))
                        } else {
                            format!("hand[{index}]")
                        }
                    } else {
                        format!("hand[{index}]")
                    }
                }
                CardSource::Stock => {
                    let self_player = state
                        .players
                        .iter()
                        .find(|player| player.id == state.self_player);
                    if let Some(player) = self_player {
                        if let Some(card) = player.stock_top {
                            if options.include_card_details {
                                format!("stock top {}", format_card(card))
                            } else {
                                String::from("stock top")
                            }
                        } else {
                            String::from("stock (empty)")
                        }
                    } else {
                        String::from("stock")
                    }
                }
                CardSource::Discard(index) => {
                    let self_player = state
                        .players
                        .iter()
                        .find(|player| player.id == state.self_player);
                    if let Some(player) = self_player {
                        let top = player
                            .discard_piles
                            .get(*index)
                            .and_then(|pile| pile.last())
                            .copied();
                        if options.include_card_details {
                            let text = top.map(format_card).unwrap_or_else(|| String::from("--"));
                            format!("discard[{index}] {text}")
                        } else {
                            format!("discard[{index}]")
                        }
                    } else {
                        format!("discard[{index}]")
                    }
                }
            };
            if options.include_build_expectation {
                format!(
                    "Play {source_desc} to build pile {} (needs {})",
                    build_pile, pile_info
                )
            } else {
                format!("Play {source_desc} to build pile {build_pile}")
            }
        }
        Action::Discard {
            hand_index,
            discard_pile,
        } => {
            let card_desc = state
                .hand
                .get(*hand_index)
                .map(|card| format_card(*card))
                .unwrap_or_else(|| String::from("--"));
            if options.include_card_details {
                format!("Discard hand[{hand_index}] {card_desc} to pile {discard_pile}")
            } else {
                format!("Discard hand[{hand_index}] to pile {discard_pile}")
            }
        }
        Action::EndTurn => String::from("End turn"),
    }
}

fn format_card(card: Card) -> String {
    match card {
        Card::Number(value) => value.to_string(),
        Card::SkipBo => String::from("SB"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::Action;
    use crate::game::GameBuilder;

    #[test]
    fn render_and_describe_include_expected_phrases() {
        let game = GameBuilder::new(2).expect("builder").build().expect("game");
        let view = game.state_view(0).expect("state view");
        let text = render_state(&view);
        assert!(text.contains("Player 0 (You)"));
        assert!(text.contains("Hand:"));
        let actions = game.legal_actions(0).expect("actions available");
        if let Some(play_action) = actions
            .iter()
            .find(|action| matches!(action, Action::Play { .. }))
        {
            let desc = describe_action(&view, play_action);
            assert!(desc.contains("build pile"));
        }
        // Ensure we can describe a discard action as well.
        let discard_action = Action::Discard {
            hand_index: 0,
            discard_pile: 0,
        };
        let discard_desc = describe_action(&view, &discard_action);
        assert!(discard_desc.contains("Discard"));
    }
}
