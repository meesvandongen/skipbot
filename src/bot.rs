use std::io::{self, Write};

use burn::tensor::backend::Backend;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::action::{Action, CardSource};
use crate::card::{Card, MAX_CARD_VALUE};
use crate::ml::{ActionSpace, PolicyNetwork, StateEncoder};
use crate::state::{GameStateView, PlayerPublicState};
use crate::visualize::{describe_action, render_state};

/// Interface for defining custom Skip-Bo bots.
pub trait Bot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action;
}

/// Baseline bot that samples uniformly from the legal action set.
pub struct RandomBot<R: Rng> {
    rng: R,
}

impl<R: Rng> RandomBot<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R: Rng> Bot for RandomBot<R> {
    fn select_action(&mut self, _state: &GameStateView, legal_actions: &[Action]) -> Action {
        legal_actions
            .choose(&mut self.rng)
            .cloned()
            .expect("at least one legal action must be available")
    }
}

/// Interactive bot that queries a human via standard input.
pub struct HumanBot {
    name: String,
}

impl HumanBot {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for HumanBot {
    fn default() -> Self {
        Self::new("Human")
    }
}

impl Bot for HumanBot {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "at least one legal action must exist"
        );
        loop {
            println!(
                "\n=== {}'s turn (player {}) ===",
                self.name, state.self_player
            );
            println!("{}", render_state(state));
            println!("Available actions:");
            for (index, action) in legal_actions.iter().enumerate() {
                println!("  [{index}] {}", describe_action(state, action));
            }
            println!("Type the action index, 'help' or 'q' to quit.");
            print!("Selection: ");
            if io::stdout().flush().is_err() {
                eprintln!("failed to flush stdout");
            }
            let mut input = String::new();
            if io::stdin().read_line(&mut input).is_err() {
                eprintln!("failed to read input");
                continue;
            }
            let trimmed = input.trim();
            if trimmed.eq_ignore_ascii_case("q") || trimmed.eq_ignore_ascii_case("quit") {
                println!("Exiting game at user's request.");
                std::process::exit(0);
            }
            if trimmed.eq_ignore_ascii_case("help") {
                println!("Enter the numeric index listed next to the action you wish to perform.");
                println!("The state summary is shown above for reference.");
                continue;
            }
            let Ok(choice) = trimmed.parse::<usize>() else {
                println!("Invalid input: '{trimmed}'. Please enter a number.");
                continue;
            };
            if let Some(action) = legal_actions.get(choice) {
                let action = action.clone();
                println!("You selected: {}", describe_action(state, &action));
                return action;
            }
            println!("Index out of range. Please choose a valid option.");
        }
    }
}

/// Rule-based bot that prioritizes progressing stock and build piles.
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
        let source_bonus = match source {
            CardSource::Stock => 10_000,
            CardSource::Discard(_) => 4_000,
            CardSource::Hand(_) => 2_000,
        };
        let value_score = Self::card_priority(card) * 60;
        let progress_bonus = (pile.cards.len() as i32) * 40;
        let closeness_bonus = (pile.next_value as i32) * 25;
        let completion_bonus = if pile.next_value == MAX_CARD_VALUE {
            1_000
        } else {
            0
        };
        let wild_bonus = if matches!(card, Card::SkipBo) { 300 } else { 0 };
        source_bonus
            + value_score
            + progress_bonus
            + closeness_bonus
            + completion_bonus
            + wild_bonus
    }

    fn score_discard(state: &GameStateView, hand_index: usize, discard_pile: usize) -> i32 {
        let Some(card) = state.hand.get(hand_index).copied() else {
            return i32::MIN / 2;
        };
        let player = Self::self_player(state);
        let existing_top = player
            .discard_tops
            .get(discard_pile)
            .and_then(|value| *value);
        let duplicate_bonus = if existing_top == Some(card) { 600 } else { 0 };
        let pile_depth = player
            .discard_counts
            .get(discard_pile)
            .copied()
            .unwrap_or_default() as i32;
        let priority = Self::card_priority(card) * 12;
        let spacing_penalty = pile_depth * 20;
        1_000 + duplicate_bonus + priority - spacing_penalty - (hand_index as i32 * 10)
    }

    fn score_action(state: &GameStateView, action: &Action) -> i32 {
        match action {
            Action::Play { source, build_pile } => Self::score_play(state, *source, *build_pile),
            Action::Discard {
                hand_index,
                discard_pile,
            } => Self::score_discard(state, *hand_index, *discard_pile),
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

/// Policy-driven bot backed by a Burn neural network.
pub struct PolicyBot<B: Backend> {
    policy: PolicyNetwork<B>,
}

impl<B: Backend> PolicyBot<B> {
    pub fn new(policy: PolicyNetwork<B>) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> &PolicyNetwork<B> {
        &self.policy
    }
}

impl<B: Backend> Bot for PolicyBot<B> {
    fn select_action(&mut self, state: &GameStateView, legal_actions: &[Action]) -> Action {
        assert!(
            !legal_actions.is_empty(),
            "policy bot requires at least one legal action"
        );
        let input = StateEncoder::encode_tensor::<B>(state);
        let logits = self.policy.forward(input).reshape([ActionSpace::MAX]);
        let values: Vec<f32> = logits
            .into_data()
            .to_vec::<f32>()
            .expect("tensor conversion");
        let mut best: Option<(f32, Action)> = None;
        for action in legal_actions {
            if let Some(index) = ActionSpace::action_index(action) {
                let value = values[index];
                match &mut best {
                    Some((best_value, best_action)) => {
                        if value > *best_value {
                            *best_value = value;
                            *best_action = action.clone();
                        }
                    }
                    None => best = Some((value, action.clone())),
                }
            }
        }
        best.map(|(_, action)| action)
            .unwrap_or_else(|| legal_actions[0].clone())
    }
}
