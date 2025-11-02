use std::error::Error;

use burn_ndarray::NdArray;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::bots::heuristic_2::Heuristic2Bot;
use crate::bots::heuristic_3::Heuristic3Bot;
use crate::bots::heuristic_4::Heuristic4Bot;
use crate::bots::heuristic_5::Heuristic5Bot;
use crate::bots::heuristic_6::Heuristic6Bot;
use crate::bots::heuristic_7::Heuristic7Bot;
use crate::bots::heuristic_8::Heuristic8Bot;
use crate::bots::heuristic_9::Heuristic9Bot;
use crate::{Bot, DEFAULT_HIDDEN, DEFAULT_STACK};
use crate::{HeuristicBot, HumanBot, PolicyBot, PolicyNetwork, RandomBot};

/// Minimal backend for policy inference used in CLI-created bots.
type PolicyBackend = NdArray<f32>;

/// Returns a normalized label for a bot spec (the head token before any ':').
pub fn label_for_spec(spec: &str) -> String {
    spec.split(':')
        .next()
        .unwrap_or(spec)
        .trim()
        .to_ascii_lowercase()
}

/// Parse a policy spec like "policy", "policy:128", or "policy:128x3" into (hidden, depth).
pub fn parse_policy_spec(spec: &str) -> Result<(usize, usize), Box<dyn Error>> {
    if let Some((_, config)) = spec.split_once(':') {
        if config.is_empty() {
            return Ok((DEFAULT_HIDDEN, DEFAULT_STACK));
        }
        if let Some((hidden, depth)) = config.split_once('x') {
            let hidden = hidden
                .parse::<usize>()
                .map_err(|_| format!("invalid hidden size in spec '{spec}'"))?;
            let depth = depth
                .parse::<usize>()
                .map_err(|_| format!("invalid depth in spec '{spec}'"))?;
            Ok((hidden, depth))
        } else {
            let hidden = config
                .parse::<usize>()
                .map_err(|_| format!("invalid hidden size in spec '{spec}'"))?;
            Ok((hidden, DEFAULT_STACK))
        }
    } else {
        Ok((DEFAULT_HIDDEN, DEFAULT_STACK))
    }
}

/// Create a bot instance from a CLI-style spec.
/// Supported specs:
/// - human[:name]
/// - random[:seed]
/// - heuristic
/// - heuristic2
/// - policy[:hidden[xdepth]]
/// - heuristic3
/// - heuristic4
/// - heuristic5
/// - heuristic6
/// - heuristic7
/// - heuristic8
/// - heuristic9
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
    } else if spec_lower.starts_with("heuristic") {
        Ok(Box::new(HeuristicBot::default()))
    } else if spec_lower.starts_with("policy") {
        let (hidden, depth) = parse_policy_spec(spec)?;
        let network = PolicyNetwork::<PolicyBackend>::new(hidden, depth);
        Ok(Box::new(PolicyBot::new(network)))
    } else {
        Err(format!("unrecognized bot spec: {spec}").into())
    }
}
