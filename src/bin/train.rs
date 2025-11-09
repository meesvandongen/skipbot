use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};

use burn::data::dataloader::Progress;
use burn::module::{AutodiffModule, Module};
use burn::optim::AdamConfig;
use burn::optim::LearningRate;
use burn::record::{BinBytesRecorder, FullPrecisionSettings, Recorder};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use burn_train::Interrupter;
use burn_train::logger::{FileMetricLogger, MetricLogger};
use burn_train::metric::{MetricEntry, NumericEntry};
use burn_train::renderer::tui::TuiMetricsRenderer;
use burn_train::renderer::{MetricState, MetricsRendererTraining, TrainingProgress};
use clap::{Parser, ValueEnum};
use rand::{Rng, RngCore, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};

use skipbot::action::Action;
use skipbot::game::GameBuilder;
use skipbot::ml::{
    ActionSpace, PolicyDataset, PolicyNetwork, PolicySample, PolicyTrainer, StateEncoder,
    TrainingEpochMetrics, TrainingLoopConfig,
};
use skipbot::{Bot, DEFAULT_HIDDEN, DEFAULT_STACK, GameError, HeuristicBot, RandomBot};

type TrainBackend = Autodiff<NdArray<f32>>;
type InferenceBackend = NdArray<f32>;
type PolicyRecord = <PolicyNetwork<InferenceBackend> as Module<InferenceBackend>>::Record;

#[derive(Parser, Debug)]
#[command(
    about = "Train Skip-Bo policy bots using the Burn framework",
    version,
    author
)]
struct TrainArgs {
    /// Number of players per game during data collection.
    #[arg(long, default_value_t = 4)]
    players: usize,
    /// Number of self-play games to collect per bot.
    #[arg(long = "games", default_value_t = 512)]
    games_per_bot: usize,
    /// Mini-batch size used during optimization.
    #[arg(long, default_value_t = 64)]
    batch_size: usize,
    /// Number of training epochs per bot.
    #[arg(long, default_value_t = 20)]
    epochs: usize,
    /// Number of policies to train.
    #[arg(long, default_value_t = 4)]
    bots: usize,
    /// Hidden layer width for the policy network.
    #[arg(long, default_value_t = DEFAULT_HIDDEN)]
    hidden: usize,
    /// Number of hidden layers (stack depth) for the policy network.
    #[arg(long, default_value_t = DEFAULT_STACK)]
    depth: usize,
    /// Learning rate passed to the Adam optimizer.
    #[arg(long, default_value_t = 1.0e-3)]
    learning_rate: f32,
    /// Fraction of the dataset to hold out for validation (0.0 - 0.5).
    #[arg(long, default_value_t = 0.1)]
    validation_split: f32,
    /// Directory where checkpoints will be written.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Exploration probability applied during data collection.
    #[arg(long, default_value_t = 0.05)]
    exploration: f32,
    /// Weight multiplier applied to moves made by the winning player.
    #[arg(long, default_value_t = 2.0)]
    winner_weight: f32,
    /// Weight multiplier applied to moves made by non-winning players.
    #[arg(long, default_value_t = 1.0)]
    runner_weight: f32,
    /// Weight multiplier applied when a game finishes without a winner.
    #[arg(long, default_value_t = 1.0)]
    draw_weight: f32,
    /// Optional cap on turns per game during collection.
    #[arg(long)]
    max_turns: Option<usize>,
    /// Master seed controlling reproducibility.
    #[arg(long, default_value_t = 0xA11C_E5EED_F00Du64)]
    seed: u64,
    /// Teacher policy used to generate training data.
    #[arg(long, value_enum, default_value_t = TeacherKind::Heuristic)]
    teacher: TeacherKind,
    /// Optional override for per-player stock size (default rules when omitted).
    #[arg(long)]
    stock_size: Option<usize>,
    /// Early stopping patience (epochs without validation improvement). Disabled when omitted.
    #[arg(long)]
    patience: Option<usize>,
    /// Resume training from a checkpoint (.bin) created by this program.
    #[arg(long)]
    resume: Option<PathBuf>,
}

#[derive(Copy, Clone, Debug, ValueEnum, Serialize, Deserialize)]
enum TeacherKind {
    Heuristic,
    Random,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PolicyMetadata {
    hidden: usize,
    depth: usize,
    learning_rate: f32,
    epochs: usize,
    batch_size: usize,
    players: usize,
    games: usize,
    seed: u64,
    dataset_seed: u64,
    train_samples: usize,
    validation_samples: usize,
    final_train_loss: f32,
    final_validation_loss: Option<f32>,
    exploration: f32,
    teacher: TeacherKind,
    winner_weight: f32,
    runner_weight: f32,
    draw_weight: f32,
    max_turns: Option<usize>,
    stock_size: Option<usize>,
    patience: Option<usize>,
    best_validation_loss: Option<f32>,
}

#[derive(Serialize, Deserialize)]
struct PolicyCheckpoint {
    metadata: PolicyMetadata,
    weights: Vec<u8>,
}

struct PendingSample {
    player: usize,
    state: [f32; skipbot::ml::STATE_FEATURES],
    mask: [f32; ActionSpace::MAX],
    target: [f32; ActionSpace::MAX],
}

fn main() {
    if let Err(err) = run() {
        eprintln!("Error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = TrainArgs::parse();
    validate_args(&args)?;
    // Load checkpoint if resuming.
    let resume_checkpoint: Option<PolicyCheckpoint> = if let Some(ref path) = args.resume {
        Some(load_checkpoint(path)?)
    } else {
        None
    };
    // Resolve output directory: explicit flag > parent of resume file > default.
    let output_dir = if let Some(dir) = args.output.clone() {
        dir
    } else if let Some(ref ckpt_path) = args.resume {
        ckpt_path.parent().unwrap_or(Path::new(".")).to_path_buf()
    } else {
        PathBuf::from("checkpoints")
    };
    fs::create_dir_all(&output_dir)?;

    let mut master_rng = StdRng::seed_from_u64(args.seed);
    // If resuming, we always train a single bot (the checkpointed one).
    let total_bots = if resume_checkpoint.is_some() {
        1
    } else {
        args.bots
    };
    for bot_index in 0..total_bots {
        // Determine dataset seed and hyperparameters when resuming.
        let dataset_seed = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.dataset_seed)
            .unwrap_or_else(|| master_rng.next_u64());
        let hidden = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.hidden)
            .unwrap_or(args.hidden);
        let depth = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.depth)
            .unwrap_or(args.depth);
        let players = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.players)
            .unwrap_or(args.players);
        let games_per_bot = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.games)
            .unwrap_or(args.games_per_bot);
        let teacher = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.teacher)
            .unwrap_or(args.teacher);
        let exploration = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.exploration)
            .unwrap_or(args.exploration);
        let winner_weight = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.winner_weight)
            .unwrap_or(args.winner_weight);
        let runner_weight = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.runner_weight)
            .unwrap_or(args.runner_weight);
        let draw_weight = resume_checkpoint
            .as_ref()
            .map(|c| c.metadata.draw_weight)
            .unwrap_or(args.draw_weight);
        let max_turns = resume_checkpoint
            .as_ref()
            .and_then(|c| c.metadata.max_turns)
            .or(args.max_turns);
        let stock_size = resume_checkpoint
            .as_ref()
            .and_then(|c| c.metadata.stock_size)
            .or(args.stock_size);
        println!(
            "\n=== Training bot {}/{} (dataset seed {:#x}) ===",
            bot_index + 1,
            total_bots,
            dataset_seed
        );
        // Build an effective args view for dataset collection using resolved values.
        let effective_args = TrainArgs {
            players,
            games_per_bot,
            batch_size: args.batch_size,
            epochs: args.epochs,
            bots: 1,
            hidden,
            depth,
            learning_rate: args.learning_rate,
            validation_split: args.validation_split,
            output: Some(output_dir.clone()),
            exploration,
            winner_weight,
            runner_weight,
            draw_weight,
            max_turns,
            seed: args.seed,
            teacher,
            stock_size,
            patience: args.patience,
            resume: args.resume.clone(),
        };
        let raw_dataset = collect_dataset(&effective_args, dataset_seed)?;
        let total_samples = raw_dataset.len();
        if total_samples == 0 {
            return Err("data collection returned an empty dataset".into());
        }

        let mut rng = StdRng::seed_from_u64(dataset_seed ^ 0x5EED_B07);
        let (mut train_dataset, validation_raw) =
            raw_dataset.split(args.validation_split, &mut rng);
        let validation_dataset = if validation_raw.is_empty() {
            None
        } else {
            Some(validation_raw)
        };
        println!(
            "  dataset split -> train: {} | validation: {} | total: {}",
            train_dataset.len(),
            validation_dataset.as_ref().map(|ds| ds.len()).unwrap_or(0),
            total_samples,
        );

        // Prepare Burn dashboard-compatible metric loggers
        // We create a per-bot run directory under the output directory.
        let run_dir = output_dir.join(format!("burn-run-bot-{:02}", bot_index + 1));
        let train_log_dir = run_dir.join("train");
        let valid_log_dir = run_dir.join("valid");
        fs::create_dir_all(&train_log_dir)?;
        fs::create_dir_all(&valid_log_dir)?;
        let mut train_logger = FileMetricLogger::new_train(&train_log_dir);
        let mut valid_logger = FileMetricLogger::new_eval(&valid_log_dir);

        // Initialize the Burn TUI renderer to visualize training progress live.
        // Note: We keep file logging active and also push updates to the TUI.
        let mut tui =
            TuiMetricsRenderer::new(Interrupter::default(), Some(args.epochs)).persistent();

        let learning_rate: LearningRate = args.learning_rate as f64;
        let optim_config = AdamConfig::new();
        let mut model = PolicyNetwork::<TrainBackend>::new(hidden, depth);
        // If resuming, load weights from checkpoint bytes.
        if let Some(ref ckpt) = resume_checkpoint {
            let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
            let record = BinBytesRecorder::<FullPrecisionSettings>::new().load::<<PolicyNetwork<
                TrainBackend,
            > as Module<TrainBackend>>::Record>(
                ckpt.weights.clone(),
                &device,
            )?;
            model = model.load_record(record);
            println!(
                "  resumed from checkpoint (hidden={}, depth={}, best_val={:?})",
                ckpt.metadata.hidden, ckpt.metadata.depth, ckpt.metadata.best_validation_loss
            );
        }
        let mut trainer = PolicyTrainer::with_config(model, optim_config, learning_rate);
        let loop_config = TrainingLoopConfig {
            epochs: args.epochs,
            batch_size: args.batch_size,
        };
        let mut training_rng = StdRng::seed_from_u64(dataset_seed ^ 0x9E37_79B9);
        // Streaming training with optional early stopping and best checkpointing
        let mut best_val: Option<f32> = resume_checkpoint
            .as_ref()
            .and_then(|c| c.metadata.best_validation_loss);
        let mut epochs_no_improve: usize = 0;
        let mut final_metrics: Option<TrainingEpochMetrics> = None;
        let train_len = train_dataset.len();
        // Closure invoked when a new best validation loss is found (passed the model reference).
        let mut on_best_checkpoint =
            |model: &PolicyNetwork<TrainBackend>, metrics: &TrainingEpochMetrics| {
                if let Some(val_loss) = metrics.validation_loss {
                    let record: PolicyRecord = model.clone().valid().into_record();
                    if let Ok(best_weights) =
                        BinBytesRecorder::<FullPrecisionSettings>::new().record(record, ())
                    {
                        let best_meta = PolicyMetadata {
                            hidden,
                            depth,
                            learning_rate: args.learning_rate,
                            epochs: metrics.epoch,
                            batch_size: args.batch_size,
                            players,
                            games: games_per_bot,
                            seed: args.seed,
                            dataset_seed,
                            train_samples: train_len,
                            validation_samples: validation_dataset
                                .as_ref()
                                .map(|ds| ds.len())
                                .unwrap_or(0),
                            final_train_loss: metrics.train_loss,
                            final_validation_loss: Some(val_loss),
                            exploration,
                            teacher,
                            winner_weight,
                            runner_weight,
                            draw_weight,
                            max_turns,
                            stock_size,
                            patience: args.patience,
                            best_validation_loss: Some(val_loss),
                        };
                        let best_checkpoint = PolicyCheckpoint {
                            metadata: best_meta,
                            weights: best_weights,
                        };
                        if let Ok(bytes) = bincode::serde::encode_to_vec(
                            &best_checkpoint,
                            bincode::config::standard(),
                        ) {
                            let filename = format!("policy-bot-{:02}-best.bin", bot_index + 1);
                            let path = output_dir.join(filename);
                            let _ = fs::write(&path, bytes);
                            println!("  best checkpoint updated -> {}", display_path(&path));
                        }
                    }
                }
            };

        let history = trainer.fit_streaming(
            &mut train_dataset,
            validation_dataset.as_ref(),
            loop_config,
            &mut training_rng,
            |metrics| {
                // Suppress stdout printing during training to avoid disrupting the TUI.
                final_metrics = Some(metrics.clone());
                if metrics.batches > 0 && metrics.samples > 0 {
                    let serialize = format!("{:.8},{}", metrics.train_loss as f64, metrics.samples);
                    let entry = MetricEntry::new(
                        "Loss".to_string().into(),
                        format!(
                            "epoch {:.6} (batches {}, samples {})",
                            metrics.train_loss, metrics.batches, metrics.samples
                        ),
                        serialize,
                    );
                    train_logger.log(&entry);
                    let num = NumericEntry::Value(metrics.train_loss as f64);
                    let train_state = MetricState::Numeric(entry.clone(), num);
                    tui.update_train(train_state);
                    let prog = TrainingProgress {
                        progress: Progress {
                            items_processed: metrics.samples,
                            items_total: train_len,
                        },
                        epoch: metrics.epoch,
                        epoch_total: args.epochs,
                        iteration: metrics.batches,
                    };
                    tui.render_train(prog);
                }
                if let Some(val_loss) = metrics.validation_loss {
                    if let Some(val_ds) = validation_dataset.as_ref() {
                        let val_samples = val_ds.len().max(1);
                        let serialize = format!("{:.8},{}", val_loss as f64, val_samples);
                        let entry = MetricEntry::new(
                            "Loss".to_string().into(),
                            format!("epoch {:.6} (samples {})", val_loss, val_samples),
                            serialize,
                        );
                        valid_logger.log(&entry);
                        let num_val = NumericEntry::Value(val_loss as f64);
                        let valid_state = MetricState::Numeric(entry.clone(), num_val);
                        tui.update_valid(valid_state);
                        let vprog = TrainingProgress {
                            progress: Progress {
                                items_processed: val_samples,
                                items_total: val_samples,
                            },
                            epoch: metrics.epoch,
                            epoch_total: args.epochs,
                            iteration: metrics.batches,
                        };
                        tui.render_valid(vprog);
                    }
                    let improved = best_val.map(|b| val_loss < b).unwrap_or(true);
                    if improved {
                        best_val = Some(val_loss);
                        epochs_no_improve = 0;
                    } else if let Some(patience) = args.patience {
                        epochs_no_improve += 1;
                        if epochs_no_improve >= patience {
                            train_logger.end_epoch(metrics.epoch);
                            return false;
                        }
                    }
                }
                train_logger.end_epoch(metrics.epoch);
                true
            },
            Some(&mut on_best_checkpoint),
        );
        let _ = tui.on_train_end(None);

        let final_train_loss = final_metrics.as_ref().map(|m| m.train_loss).unwrap_or(0.0);
        let final_validation_loss = final_metrics.as_ref().and_then(|m| m.validation_loss);

        let record: PolicyRecord = trainer.model().clone().valid().into_record();
        let weights = BinBytesRecorder::<FullPrecisionSettings>::new()
            .record(record, ())
            .map_err(|err| -> Box<dyn Error> { Box::new(err) })?;
        let metadata = PolicyMetadata {
            hidden,
            depth,
            learning_rate: args.learning_rate,
            epochs: args.epochs,
            batch_size: args.batch_size,
            players,
            games: games_per_bot,
            seed: args.seed,
            dataset_seed,
            train_samples: train_len,
            validation_samples: validation_dataset.as_ref().map(|ds| ds.len()).unwrap_or(0),
            final_train_loss,
            final_validation_loss,
            exploration,
            teacher,
            winner_weight,
            runner_weight,
            draw_weight,
            max_turns,
            stock_size,
            patience: args.patience,
            best_validation_loss: best_val,
        };
        let checkpoint = PolicyCheckpoint { metadata, weights };
        let bytes = bincode::serde::encode_to_vec(&checkpoint, bincode::config::standard())?;
        let filename = format!("policy-bot-{:02}.bin", bot_index + 1);
        let path = output_dir.join(filename);
        fs::write(&path, bytes)?;
        println!("  checkpoint saved -> {}", display_path(&path));
    }
    Ok(())
}

fn validate_args(args: &TrainArgs) -> Result<(), Box<dyn Error>> {
    if !(2..=6).contains(&args.players) {
        return Err("players must be between 2 and 6".into());
    }
    if args.games_per_bot == 0 {
        return Err("games per bot must be positive".into());
    }
    if args.batch_size == 0 {
        return Err("batch size must be positive".into());
    }
    if !(0.0..1.0).contains(&args.validation_split) {
        return Err("validation split must be in [0, 1)".into());
    }
    if !(0.0..=1.0).contains(&args.exploration) {
        return Err("exploration rate must be between 0 and 1".into());
    }
    if args.learning_rate <= 0.0 {
        return Err("learning rate must be positive".into());
    }
    if let Some(stock) = args.stock_size {
        if stock == 0 {
            return Err("stock-size must be positive".into());
        }
    }
    Ok(())
}

#[allow(dead_code)]
fn log_epoch(_metrics: &TrainingEpochMetrics) {
    // Intentionally suppressed to keep the TUI stable during training.
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn load_checkpoint(path: &Path) -> Result<PolicyCheckpoint, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    let (ckpt, _): (PolicyCheckpoint, usize) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard())?;
    Ok(ckpt)
}

fn collect_dataset(args: &TrainArgs, dataset_seed: u64) -> Result<PolicyDataset, GameError> {
    let mut rng = StdRng::seed_from_u64(dataset_seed);
    let mut dataset = PolicyDataset::new();
    for game_index in 0..args.games_per_bot {
        let game_seed = rng.next_u64();
        let mut builder = GameBuilder::new(args.players)?.with_seed(game_seed);
        if let Some(stock) = args.stock_size {
            builder = builder.with_stock_size(stock);
        }
        let mut game = builder.build()?;
        let mut bots = build_teacher_bots(args.teacher, args.players, dataset_seed, &mut rng);
        let mut trajectory: Vec<PendingSample> = Vec::new();
        let mut turns = 0usize;
        while !game.is_finished() {
            if let Some(limit) = args.max_turns {
                if turns >= limit {
                    break;
                }
            }
            let current = game.current_player();
            let state = game.state_view(current)?;
            let legal = game.legal_actions(current)?;
            if legal.is_empty() {
                break;
            }
            let action = select_teacher_action(
                bots[current].as_mut(),
                &state,
                &legal,
                &mut rng,
                args.exploration,
            );
            let features = StateEncoder::encode(&state);
            let mask = ActionSpace::mask(&legal);
            let mut target = [0.0f32; ActionSpace::MAX];
            if let Some(index) = ActionSpace::action_index(&action) {
                target[index] = 1.0;
            }
            trajectory.push(PendingSample {
                player: current,
                state: features,
                mask,
                target,
            });
            game.apply_action(current, action)?;
            turns += 1;
        }
        let winner = game.winner();
        for sample in trajectory {
            let weight = match winner {
                Some(id) if id == sample.player => args.winner_weight,
                Some(_) => args.runner_weight,
                None => args.draw_weight,
            };
            let weight = weight.max(0.0);
            dataset.push(PolicySample::from_components(
                sample.state,
                sample.mask,
                sample.target,
                weight,
            ));
        }
        if (game_index + 1) % 50 == 0 {
            println!(
                "  collected games: {}/{} (current dataset size: {})",
                game_index + 1,
                args.games_per_bot,
                dataset.len()
            );
        }
    }
    Ok(dataset)
}

fn build_teacher_bots(
    kind: TeacherKind,
    count: usize,
    base_seed: u64,
    rng: &mut StdRng,
) -> Vec<Box<dyn Bot>> {
    (0..count)
        .map(|idx| match kind {
            TeacherKind::Heuristic => Box::new(HeuristicBot::default()) as Box<dyn Bot>,
            TeacherKind::Random => {
                let seed = rng.next_u64() ^ base_seed ^ ((idx as u64 + 1) * 0x9E37_79B9);
                Box::new(RandomBot::new(StdRng::seed_from_u64(seed))) as Box<dyn Bot>
            }
        })
        .collect()
}

fn select_teacher_action(
    bot: &mut dyn Bot,
    state: &skipbot::state::GameStateView,
    legal: &[Action],
    rng: &mut StdRng,
    exploration: f32,
) -> Action {
    if rng.gen_range(0.0..1.0) < exploration {
        legal[rng.gen_range(0..legal.len())].clone()
    } else {
        bot.select_action(state, legal)
    }
}
