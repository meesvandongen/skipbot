use std::error::Error;

use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::Bot;
use crate::bots::heuristic_2::Heuristic2Bot;
use crate::bots::heuristic_3::Heuristic3Bot;
use crate::bots::heuristic_4::Heuristic4Bot;
use crate::bots::heuristic_5::Heuristic5Bot;
use crate::bots::heuristic_6::Heuristic6Bot;
use crate::bots::heuristic_7::Heuristic7Bot;
use crate::bots::heuristic_8::Heuristic8Bot;
use crate::bots::heuristic_9::Heuristic9Bot;
use crate::bots::heuristic_10::Heuristic10Bot;
use crate::bots::heuristic_11::Heuristic11Bot;
use crate::bots::heuristic_12::Heuristic12Bot;
use crate::bots::heuristic_13::Heuristic13Bot;
use crate::bots::heuristic_14::Heuristic14Bot;
use crate::bots::heuristic_15::Heuristic15Bot;
use crate::{HeuristicBot, HumanBot, RandomBot};

/// Returns a normalized label for a bot spec (the head token before any ':').
pub fn label_for_spec(spec: &str) -> String {
    spec.split(':')
        .next()
        .unwrap_or(spec)
        .trim()
        .to_ascii_lowercase()
}

/// Create a bot instance from a CLI-style spec.
/// Supported specs:
/// - human[:name]
/// - random[:seed]
/// - heuristic
/// - heuristic2
/// - heuristic3
/// - heuristic4
/// - heuristic5
/// - heuristic6
/// - heuristic7
/// - heuristic8
/// - heuristic9
/// - heuristic10
/// - heuristic11
/// - heuristic12
/// - heuristic13
/// - heuristic14
/// - heuristic15
pub fn create_bot_from_spec(
    spec: &str,
    index: usize,
    seed: u64,
) -> Result<Box<dyn Bot>, Box<dyn Error>> {
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
    } else if spec_lower.starts_with("heuristic2") {
        Ok(Box::new(Heuristic2Bot::default()))
    } else if spec_lower.starts_with("heuristic3") {
        Ok(Box::new(Heuristic3Bot::default()))
    } else if spec_lower.starts_with("heuristic4") {
        Ok(Box::new(Heuristic4Bot::default()))
    } else if spec_lower.starts_with("heuristic5") {
        Ok(Box::new(Heuristic5Bot::default()))
    } else if spec_lower.starts_with("heuristic6") {
        Ok(Box::new(Heuristic6Bot::default()))
    } else if spec_lower.starts_with("heuristic7") {
        Ok(Box::new(Heuristic7Bot::default()))
    } else if spec_lower.starts_with("heuristic8") {
        Ok(Box::new(Heuristic8Bot::default()))
    } else if spec_lower.starts_with("heuristic9") {
        Ok(Box::new(Heuristic9Bot::default()))
    } else if spec_lower.starts_with("heuristic10") {
        Ok(Box::new(Heuristic10Bot::default()))
    } else if spec_lower.starts_with("heuristic11") {
        Ok(Box::new(Heuristic11Bot::default()))
    } else if spec_lower.starts_with("heuristic12") {
        Ok(Box::new(Heuristic12Bot::default()))
    } else if spec_lower.starts_with("heuristic13") {
        Ok(Box::new(Heuristic13Bot::default()))
    } else if spec_lower.starts_with("heuristic14") {
        Ok(Box::new(Heuristic14Bot::default()))
    } else if spec_lower.starts_with("heuristic15") {
        Ok(Box::new(Heuristic15Bot::default()))
    } else if spec_lower.starts_with("heuristic") {
        Ok(Box::new(HeuristicBot::default()))
    } else {
        Err(format!("unrecognized bot spec: {spec}").into())
    }
}
