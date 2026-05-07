use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use std::process;
use std::time::Instant;

use clap::{ArgAction, Parser, ValueEnum};
use plotters::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
// no need to import Shift when drawing inline

use skipbot::winner_points;
use skipbot::{Bot, Game, GameError};
use skipbot::{create_bot_from_spec, label_for_spec};

/// Default base seed for deterministic runs.
const DEFAULT_SEED: u64 = 0xC0FFEE_u64 << 32 | 0x5EED_u64;

/// Output format for the generated chart. Currently only PNG is supported.
#[derive(Clone, Debug, ValueEnum)]
enum ChartFormat {
    Png,
}

impl ChartFormat {
    fn from_path(path: &PathBuf) -> Option<Self> {
        match path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
        {
            Some(ext) if ext == "png" => Some(Self::Png),
            _ => None,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "winrate",
    about = "Run multiple games and plot per-bot win rates."
)]
struct Args {
    /// Number of games to simulate
    #[arg(short = 'g', long = "games", default_value_t = 200)]
    games: usize,

    /// Base RNG seed (deck + bot RNGs will be derived deterministically)
    #[arg(short = 's', long = "seed", default_value_t = DEFAULT_SEED)]
    seed: u64,

    /// Output chart file (png or svg)
    #[arg(short = 'o', long = "out", default_value = "winrates.png")]
    out: PathBuf,

    /// Explicit output format (inferred from --out when omitted)
    #[arg(long = "format", value_enum)]
    format: Option<ChartFormat>,

    /// Show a textual summary only (no chart)
    #[arg(long = "no-chart", action = ArgAction::SetTrue)]
    no_chart: bool,

    /// Safety cap on turns per game; games exceeding this are aborted (not counted as a win)
    #[arg(long = "max-turns", default_value_t = 2000)]
    max_turns: usize,

    /// Optional override for per-player stock size (default rules when omitted).
    /// Useful to shorten games for quick benchmarking.
    #[arg(long = "stock-size")]
    stock_size: Option<usize>,

    /// Player bot specs: e.g., heuristic random (2-6 total)
    bots: Vec<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("Error: {err}");
        process::exit(1);
    }
}

fn run(args: Args) -> Result<(), Box<dyn Error>> {
    if args.bots.is_empty() {
        return Err("please provide between 2 and 6 bot specs (e.g., heuristic random)".into());
    }
    if args.bots.len() < 2 || args.bots.len() > 6 {
        return Err(format!(
            "expected between 2 and 6 players, received {}",
            args.bots.len()
        )
        .into());
    }

    // Disallow human in batch sims; it would block waiting for input.
    if args
        .bots
        .iter()
        .any(|s| s.to_ascii_lowercase().starts_with("human"))
    {
        return Err("human players are not supported in winrate runs".into());
    }

    if let Some(stock) = args.stock_size {
        if stock == 0 {
            return Err("stock-size must be positive".into());
        }
    }

    // Aggregate counts across all games.
    let mut wins_per_label: HashMap<String, usize> = HashMap::new();
    let mut seats_per_label: HashMap<String, usize> = HashMap::new();
    let mut points_per_label: HashMap<String, u64> = HashMap::new();
    let mut aborted_games: usize = 0;

    // Decision-time accounting per bot label.
    // Accumulates total nanoseconds spent inside select_action and counts of decisions.
    let mut decision_time_ns: HashMap<String, u128> = HashMap::new();
    let mut decision_counts: HashMap<String, usize> = HashMap::new();

    let base_seed = args.seed;
    let players_per_game = args.bots.len();

    // Precompute labels for specs to avoid recomputing.
    let labels_for_spec: Vec<String> = args.bots.iter().map(|s| label_for_spec(s)).collect();

    for game_idx in 0..args.games {
        // Permute seating each game for fairness.
        let mut indices: Vec<usize> = (0..players_per_game).collect();
        let mut seat_rng = StdRng::seed_from_u64(base_seed ^ 0x9E37_79B9 ^ (game_idx as u64));
        indices.shuffle(&mut seat_rng);

        // Create game with mixed seed.
        let deck_seed = mix_seed(base_seed, game_idx as u64, 0x5EED_15);
        let mut builder = Game::builder(players_per_game)?.with_seed(deck_seed);
        if let Some(stock) = args.stock_size {
            builder = builder.with_stock_size(stock);
        }
        let mut game = builder.build()?;

        // Build and seat bots.
        let mut bots: Vec<Box<dyn Bot>> = Vec::with_capacity(players_per_game);
        let mut labels: Vec<String> = Vec::with_capacity(players_per_game);
        for (seat, src_idx) in indices.iter().enumerate() {
            let spec = &args.bots[*src_idx];
            let label = labels_for_spec[*src_idx].clone();
            let bot_seed = mix_seed(base_seed, game_idx as u64, seat as u64);
            let bot = create_bot(spec, seat, bot_seed)?;
            bots.push(bot);
            labels.push(label);
        }

        // Increment seat counts per label for this game.
        for label in &labels {
            *seats_per_label.entry(label.clone()).or_default() += 1;
        }

        // Run the game to completion (or until max turn cap).
        let mut turns = 0usize;
        loop {
            if game.is_finished() {
                break;
            }
            if turns >= args.max_turns {
                break;
            }
            let current = game.current_player();
            let state = game.state_view(current)?;
            let legal = game.legal_actions(current)?;
            if legal.is_empty() {
                return Err(GameError::InvalidConfiguration("no legal actions available").into());
            }
            // Time the bot's decision for this move.
            let label_for_current = labels[current].clone();
            let t0 = Instant::now();
            let action = bots[current].select_action(&state, &legal);
            let dt = t0.elapsed();
            *decision_time_ns
                .entry(label_for_current.clone())
                .or_default() += dt.as_nanos();
            *decision_counts.entry(label_for_current).or_default() += 1;
            game.apply_action(current, action)?;
            turns += 1;
        }

        if let Some(winner) = game.winner() {
            let label = labels[winner].clone();
            *wins_per_label.entry(label.clone()).or_default() += 1;

            // Compute scoring: 25 points + 5 points per card in opponents' stock piles
            // Use a final state view from the winner's perspective.
            let view = game.state_view(winner)?;
            let pts = winner_points(&view, winner) as u64;
            *points_per_label.entry(label).or_default() += pts;
        } else {
            aborted_games += 1;
        }
    }

    // Compute per-seat win probability per label.
    let mut results: Vec<(String, f64, usize, usize)> = Vec::new();
    for (label, &wins) in &wins_per_label {
        let seats = *seats_per_label.get(label).unwrap_or(&0);
        let rate = if seats > 0 {
            wins as f64 / seats as f64
        } else {
            0.0
        };
        results.push((label.clone(), rate, wins, seats));
    }

    // Include labels that never won so they appear with 0%.
    for (label, &seats) in &seats_per_label {
        if !wins_per_label.contains_key(label) {
            results.push((label.clone(), 0.0, 0, seats));
        }
    }

    // Sort by rate desc, then by label.
    results.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    // Print textual summary.
    println!("Win rates (per-seat) with scoring:");
    for (label, rate, wins, seats) in &results {
        let total_points = *points_per_label.get(label).unwrap_or(&0u64);
        let avg_points = if *seats > 0 {
            total_points as f64 / (*seats as f64)
        } else {
            0.0
        };
        println!(
            "  {label:<12}  {wins}/{seats}  ({:.2}%)   avg pts: {:>6.2}   total pts: {}",
            rate * 100.0,
            avg_points,
            total_points
        );
    }
    if aborted_games > 0 {
        println!("\nNote: {aborted_games} game(s) ended without a winner (draws or timeouts).");
    }

    if !args.no_chart {
        let format = args
            .format
            .or_else(|| ChartFormat::from_path(&args.out))
            .unwrap_or(ChartFormat::Png);
        if !matches!(format, ChartFormat::Png) {
            return Err("only PNG output is supported currently; use --out with .png".into());
        }
        render_bar_chart(&args.out, &results)?;
        println!("\nChart written to {}", args.out.display());
    }

    // Print timing summary (per bot label)
    if !decision_counts.is_empty() {
        println!("\nDecision time (per bot label):");
        // Follow the same ordering as win-rate results when possible; otherwise fall back to map order.
        let mut printed: HashMap<String, bool> = HashMap::new();
        for (label, _rate, _wins, _seats) in &results {
            if let Some(&count) = decision_counts.get(label) {
                let total_ns = *decision_time_ns.get(label).unwrap_or(&0u128);
                let total_ms = (total_ns as f64) / 1.0e6;
                let avg_ms = if count > 0 {
                    total_ms / (count as f64)
                } else {
                    0.0
                };
                println!(
                    "  {label:<12}  decisions: {count:<7}  total: {total_ms:.3} ms  avg: {avg_ms:.3} ms"
                );
                printed.insert(label.clone(), true);
            }
        }
        // Print any labels that didn't appear in results (shouldn't happen, but safe).
        for (label, &count) in &decision_counts {
            if printed.get(label).copied().unwrap_or(false) {
                continue;
            }
            let total_ns = *decision_time_ns.get(label).unwrap_or(&0u128);
            let total_ms = (total_ns as f64) / 1.0e6;
            let avg_ms = if count > 0 {
                total_ms / (count as f64)
            } else {
                0.0
            };
            println!(
                "  {label:<12}  decisions: {count:<7}  total: {total_ms:.3} ms  avg: {avg_ms:.3} ms"
            );
        }
    }

    Ok(())
}

fn mix_seed(base: u64, a: u64, b: u64) -> u64 {
    // Simple reversible mixer (xorshift-like mix).
    let mut z =
        base ^ (a.wrapping_mul(0x9E37_79B97F4A7C15)) ^ (b.wrapping_mul(0xBF58_476D1CE4E5B9));
    z ^= z >> 12;
    z ^= z << 25;
    z ^= z >> 27;
    z
}

fn create_bot(spec: &str, index: usize, seed: u64) -> Result<Box<dyn Bot>, Box<dyn Error>> {
    // Centralized creation ensures parity with simulate CLI.
    create_bot_from_spec(spec, index, seed)
}

fn render_bar_chart(
    out: &PathBuf,
    data: &[(String, f64, usize, usize)],
) -> Result<(), Box<dyn Error>> {
    // Prepare values and labels
    let labels: Vec<String> = data.iter().map(|(l, _, _, _)| l.clone()).collect();
    let values: Vec<f64> = data.iter().map(|(_, r, _, _)| r * 100.0).collect();
    let max_value = values
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(100.0_f64.min(100.0));

    let root = BitMapBackend::new(out, (1000, 600)).into_drawing_area();
    root.fill(&WHITE).map_err(|e| format!("{e}"))?;

    // Build chart
    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Skip-Bo Bot Win Rates (per-seat)",
            ("sans-serif", 28).into_font(),
        )
        .margin(20)
        .x_label_area_size(50)
        .y_label_area_size(60)
        .build_cartesian_2d(0..labels.len(), 0.0f64..max_value.max(10.0))
        .map_err(|e| format!("{e}"))?;

    chart
        .configure_mesh()
        .y_desc("Win rate (%)")
        .x_desc("Bot type")
        .x_labels(labels.len())
        .x_label_formatter(&|idx| {
            if *idx < labels.len() {
                labels[*idx].clone()
            } else {
                idx.to_string()
            }
        })
        .y_label_formatter(&|v| format!("{v:.0}"))
        .light_line_style(&WHITE.mix(0.0))
        .draw()
        .map_err(|e| format!("{e}"))?;

    // Bars
    for (i, value) in values.iter().enumerate() {
        let rect = Rectangle::new([(i, 0.0), (i, *value)], BLUE.filled());
        chart
            .draw_series(std::iter::once(rect))
            .map_err(|e| format!("{e}"))?;
    }

    root.present().map_err(|e| format!("{e}"))?;
    Ok(())
}
