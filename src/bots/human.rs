use std::io::{self, Write};

use crate::action::Action;
use crate::bot::Bot;
use crate::state::GameStateView;
use crate::visualize::{describe_action, render_state};

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
