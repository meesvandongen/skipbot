use std::env;
use std::error::Error;
use std::process;

use rand::SeedableRng;
use rand::rngs::StdRng;

use skipbot::{Bot, Game, GameError, HumanBot, RandomBot, describe_action, render_state};

const DEFAULT_SEED: u64 = 0xDEC0_1DED_5EED_F00D;

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let mut args = env::args().skip(1);
    let mut visualize = false;
    let mut seed = DEFAULT_SEED;
    let mut max_turns: Option<usize> = None;
    let mut bot_specs: Vec<String> = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--visualize" => visualize = true,
            "--seed" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--seed requires a value".to_string())?;
                seed = value
                    .parse::<u64>()
                    .map_err(|_| format!("invalid seed value: {value}"))?;
            }
            "--max-turns" => {
                let value = args
                    .next()
                    .ok_or_else(|| "--max-turns requires a value".to_string())?;
                max_turns = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| format!("invalid max-turns value: {value}"))?,
                );
            }
            "--help" => {
                print_usage();
                return Ok(());
            }
            other => bot_specs.push(other.to_string()),
        }
    }

    if bot_specs.is_empty() {
        bot_specs = vec![String::from("human"), String::from("random")];
    }
    if bot_specs.len() < 2 || bot_specs.len() > 6 {
        return Err(format!(
            "expected between 2 and 6 players, received {}",
            bot_specs.len()
        )
        .into());
    }

    let num_players = bot_specs.len();
    let builder = Game::builder(num_players)?.with_seed(seed);
    let mut game = builder.build()?;

    let mut bots: Vec<Box<dyn Bot>> = Vec::with_capacity(num_players);
    for (index, spec) in bot_specs.iter().enumerate() {
        let bot = create_bot(spec, index, seed)?;
        bots.push(bot);
    }

    println!("Starting Skip-Bo simulation with {num_players} players.\n");
    let mut turns = 0usize;
    loop {
        if game.is_finished() {
            break;
        }
        if let Some(limit) = max_turns {
            if turns >= limit {
                println!("Max turn limit {limit} reached. Stopping simulation.");
                break;
            }
        }
        let current = game.current_player();
        let state = game.state_view(current)?;
        let legal_actions = game.legal_actions(current)?;
        if legal_actions.is_empty() {
            return Err(GameError::InvalidConfiguration(
                "no legal actions available for current player",
            )
            .into());
        }
        if visualize {
            println!("{}", render_state(&state));
        }
        let action = bots[current].select_action(&state, &legal_actions);
        if visualize {
            println!("Chosen action: {}\n", describe_action(&state, &action));
        }
        game.apply_action(current, action)?;
        turns += 1;
    }

    if let Some(winner) = game.winner() {
        println!("Game finished. Winner: Player {winner}.");
    } else {
        println!("Simulation stopped before completion.");
    }

    Ok(())
}

fn create_bot(spec: &str, index: usize, seed: u64) -> Result<Box<dyn Bot>, Box<dyn Error>> {
    let spec_lower = spec.to_ascii_lowercase();
    if spec_lower.starts_with("human") {
        let name = spec
            .split_once(':')
            .map(|(_, name)| name.trim().to_string());
        let name = name.unwrap_or_else(|| format!("Human {index}"));
        Ok(Box::new(HumanBot::new(name)))
    } else if spec_lower.starts_with("random") {
        let custom_seed = spec
            .split_once(':')
            .and_then(|(_, value)| value.parse::<u64>().ok())
            .unwrap_or(seed ^ ((index as u64 + 1) * 0x9E37_79B9));
        Ok(Box::new(RandomBot::new(StdRng::seed_from_u64(custom_seed))))
    } else {
        Err(format!("unrecognized bot spec: {spec}").into())
    }
}

fn print_usage() {
    println!("Usage: simulate [OPTIONS] [BOT ...]");
    println!("  --visualize           Show the game state and chosen actions each turn");
    println!("  --seed <u64>          Seed for shuffling (default: {DEFAULT_SEED:#x})");
    println!("  --max-turns <usize>   Stop after the specified number of turns");
    println!("  --help                Show this help message");
    println!("Bot entries (2-6 total):");
    println!("  human[:name]          Interactive human-controlled player");
    println!("  random[:seed]         Random bot with optional per-bot seed");
    println!("If no bots are provided, defaults to one human and one random bot.");
}
