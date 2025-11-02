use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{Adam, AdamConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use rand::Rng;
use rand::seq::SliceRandom;

use super::encoding::{ActionSpace, STATE_FEATURES, StateEncoder};
use super::policy::PolicyNetwork;
use crate::action::Action;
use crate::state::GameStateView;

const MIN_SAMPLE_WEIGHT: f32 = 1.0e-8;

#[derive(Clone, Debug)]
pub struct PolicySample {
    pub state: [f32; STATE_FEATURES],
    pub target: [f32; ActionSpace::MAX],
    pub mask: [f32; ActionSpace::MAX],
    pub weight: f32,
}

impl PolicySample {
    pub fn from_components(
        state: [f32; STATE_FEATURES],
        mask: [f32; ActionSpace::MAX],
        target: [f32; ActionSpace::MAX],
        weight: f32,
    ) -> Self {
        let clamped_weight = if weight.is_finite() {
            weight.max(MIN_SAMPLE_WEIGHT)
        } else {
            MIN_SAMPLE_WEIGHT
        };
        Self {
            state,
            target,
            mask,
            weight: clamped_weight,
        }
    }

    pub fn from_transition(
        state: &GameStateView,
        legal_actions: &[Action],
        chosen_actions: &[Action],
        weight: f32,
    ) -> Self {
        let encoded = StateEncoder::encode(state);
        let mask = ActionSpace::mask(legal_actions);
        let indices: Vec<usize> = chosen_actions
            .iter()
            .filter_map(ActionSpace::action_index)
            .collect();
        let target = ActionSpace::targets_from_indices(&indices);
        Self::from_components(encoded, mask, target, weight)
    }
}

#[derive(Default, Clone, Debug)]
pub struct PolicyDataset {
    samples: Vec<PolicySample>,
}

impl PolicyDataset {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn from_samples(samples: Vec<PolicySample>) -> Self {
        Self { samples }
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn total_weight(&self) -> f32 {
        self.samples.iter().map(|sample| sample.weight).sum()
    }

    pub fn samples(&self) -> &[PolicySample] {
        &self.samples
    }

    pub fn push(&mut self, sample: PolicySample) {
        if sample.weight.is_finite() && sample.weight > 0.0 {
            self.samples.push(sample);
        }
    }

    pub fn extend<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = PolicySample>,
    {
        for sample in iter {
            self.push(sample);
        }
    }

    pub fn shuffle<R: Rng>(&mut self, rng: &mut R) {
        self.samples.shuffle(rng);
    }

    pub fn batches(&self, batch_size: usize) -> std::slice::Chunks<'_, PolicySample> {
        let size = batch_size.max(1);
        self.samples.chunks(size)
    }

    pub fn split(mut self, validation_fraction: f32, rng: &mut impl Rng) -> (Self, Self) {
        if self.samples.len() < 2 || validation_fraction <= 0.0 {
            return (self, Self::default());
        }
        let mut fraction = validation_fraction.clamp(0.0, 0.9);
        if fraction == 0.0 {
            fraction = 0.0;
        }
        self.samples.shuffle(rng);
        let total = self.samples.len();
        let mut validation_size = ((total as f32) * fraction).round() as usize;
        validation_size = validation_size.clamp(1, total - 1);
        let split_index = total - validation_size;
        let validation_samples = self.samples.split_off(split_index);
        let train_samples = self.samples;
        (
            PolicyDataset {
                samples: train_samples,
            },
            PolicyDataset {
                samples: validation_samples,
            },
        )
    }
}

impl From<Vec<PolicySample>> for PolicyDataset {
    fn from(value: Vec<PolicySample>) -> Self {
        Self::from_samples(value)
    }
}

#[derive(Debug)]
pub struct PolicyBatch<B: AutodiffBackend> {
    pub states: Tensor<B, 2>,
    pub targets: Tensor<B, 2>,
    pub masks: Tensor<B, 2>,
    pub weights: Tensor<B, 2>,
}

impl<B: AutodiffBackend> PolicyBatch<B> {
    pub fn new(states: Tensor<B, 2>, targets: Tensor<B, 2>, masks: Tensor<B, 2>) -> Self {
        let batch = states.shape().dims[0];
        let weights = Tensor::<B, 2>::ones([batch, 1], &B::Device::default());
        Self {
            states,
            targets,
            masks,
            weights,
        }
    }

    pub fn with_weights(
        states: Tensor<B, 2>,
        targets: Tensor<B, 2>,
        masks: Tensor<B, 2>,
        weights: Tensor<B, 2>,
    ) -> Self {
        Self {
            states,
            targets,
            masks,
            weights,
        }
    }

    pub fn from_samples(samples: &[PolicySample]) -> Self {
        assert!(
            !samples.is_empty(),
            "cannot construct a policy batch from an empty sample slice"
        );
        let batch_size = samples.len();
        let mut states = Vec::with_capacity(batch_size * STATE_FEATURES);
        let mut targets = Vec::with_capacity(batch_size * ActionSpace::MAX);
        let mut masks = Vec::with_capacity(batch_size * ActionSpace::MAX);
        let mut weights = Vec::with_capacity(batch_size);
        for sample in samples {
            states.extend_from_slice(&sample.state);
            targets.extend_from_slice(&sample.target);
            masks.extend_from_slice(&sample.mask);
            weights.push(sample.weight);
        }
        let device = B::Device::default();
        let states_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(states, [batch_size, STATE_FEATURES]),
            &device,
        );
        let targets_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(targets, [batch_size, ActionSpace::MAX]),
            &device,
        );
        let masks_tensor = Tensor::<B, 2>::from_data(
            TensorData::new(masks, [batch_size, ActionSpace::MAX]),
            &device,
        );
        let weights_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(weights, [batch_size, 1]), &device);
        Self::with_weights(states_tensor, targets_tensor, masks_tensor, weights_tensor)
    }

    pub fn sample_count(&self) -> usize {
        self.states.shape().dims[0]
    }

    pub fn weight_sum(&self) -> f32 {
        self.weights
            .clone()
            .detach()
            .into_data()
            .to_vec::<f32>()
            .map(|values| values.into_iter().sum())
            .unwrap_or_default()
    }
}

#[derive(Clone, Debug)]
pub struct TrainingLoopConfig {
    pub epochs: usize,
    pub batch_size: usize,
}

#[derive(Clone, Debug)]
pub struct TrainingEpochMetrics {
    pub epoch: usize,
    pub train_loss: f32,
    pub validation_loss: Option<f32>,
    pub batches: usize,
    pub samples: usize,
}

pub struct PolicyTrainer<B: AutodiffBackend> {
    model: PolicyNetwork<B>,
    optimizer: OptimizerAdaptor<Adam, PolicyNetwork<B>, B>,
    learning_rate: LearningRate,
    step: usize,
}

impl<B: AutodiffBackend> PolicyTrainer<B> {
    pub fn new(
        model: PolicyNetwork<B>,
        optimizer: OptimizerAdaptor<Adam, PolicyNetwork<B>, B>,
        learning_rate: LearningRate,
    ) -> Self {
        Self {
            model,
            optimizer,
            learning_rate,
            step: 0,
        }
    }

    pub fn with_config(
        model: PolicyNetwork<B>,
        config: AdamConfig,
        learning_rate: LearningRate,
    ) -> Self {
        let optimizer = config.init();
        Self::new(model, optimizer, learning_rate)
    }

    pub fn model(&self) -> &PolicyNetwork<B> {
        &self.model
    }

    pub fn optimizer(&self) -> &OptimizerAdaptor<Adam, PolicyNetwork<B>, B> {
        &self.optimizer
    }

    pub fn train_step(&mut self, batch: PolicyBatch<B>) -> f32 {
        let (loss_sum, weight_sum) = Self::loss_components(&self.model, &batch);
        let loss = loss_sum.clone() / weight_sum;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let model = self.model.clone();
        self.model = self.optimizer.step(self.learning_rate, model, grads);
        self.step += 1;
        Self::tensor_to_f32(loss)
    }

    pub fn evaluate(&self, dataset: &PolicyDataset, batch_size: usize) -> f32 {
        if dataset.is_empty() {
            return 0.0;
        }
        let mut total_loss = 0.0;
        let mut total_weight = 0.0;
        for chunk in dataset.batches(batch_size) {
            if chunk.is_empty() {
                continue;
            }
            let batch = PolicyBatch::<B>::from_samples(chunk);
            let (loss_sum, weight_sum) = Self::loss_components(&self.model, &batch);
            let loss_scalar = Self::tensor_to_f32(loss_sum);
            let weight_scalar = Self::tensor_to_f32(weight_sum);
            total_loss += loss_scalar;
            total_weight += weight_scalar;
        }
        if total_weight > 0.0 {
            total_loss / total_weight
        } else {
            0.0
        }
    }

    pub fn fit<R: Rng>(
        &mut self,
        train: &mut PolicyDataset,
        validation: Option<&PolicyDataset>,
        config: TrainingLoopConfig,
        rng: &mut R,
    ) -> Vec<TrainingEpochMetrics> {
        assert!(config.batch_size > 0, "batch size must be positive");
        let mut history = Vec::with_capacity(config.epochs);
        for epoch in 0..config.epochs {
            train.shuffle(rng);
            let mut weighted_loss = 0.0;
            let mut weight_sum = 0.0;
            let mut batches = 0usize;
            let mut samples = 0usize;
            for chunk in train.batches(config.batch_size) {
                if chunk.is_empty() {
                    continue;
                }
                let batch = PolicyBatch::<B>::from_samples(chunk);
                let batch_weight = batch.weight_sum();
                if batch_weight <= 0.0 {
                    continue;
                }
                let loss = self.train_step(batch);
                weighted_loss += loss * batch_weight;
                weight_sum += batch_weight;
                batches += 1;
                samples += chunk.len();
            }
            let train_loss = if weight_sum > 0.0 {
                weighted_loss / weight_sum
            } else {
                0.0
            };
            let validation_loss = validation.map(|set| self.evaluate(set, config.batch_size));
            history.push(TrainingEpochMetrics {
                epoch: epoch + 1,
                train_loss,
                validation_loss,
                batches,
                samples,
            });
        }
        history
    }

    fn loss_components(
        model: &PolicyNetwork<B>,
        batch: &PolicyBatch<B>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let logits = model.forward(batch.states.clone());
        let masked_logits = logits + batch.masks.clone();
        let log_probs = log_softmax(masked_logits, 1);
        let cross_entropy = -(batch.targets.clone() * log_probs).sum_dim(1);
        let batch_size = batch.sample_count();
        let cross_entropy = cross_entropy.reshape([batch_size, 1]);
        let weighted = cross_entropy * batch.weights.clone();
        let loss_sum = weighted.sum();
        let weight_sum = batch.weights.clone().sum();
        (loss_sum, weight_sum)
    }

    fn tensor_to_f32(tensor: Tensor<B, 1>) -> f32 {
        tensor
            .detach()
            .into_data()
            .to_vec::<f32>()
            .map(|mut values| values.pop().unwrap_or_default())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use crate::action::Action;
    use crate::game::GameBuilder;

    type Backend = Autodiff<NdArray<f32>>;

    #[test]
    fn trainer_produces_finite_loss() {
        let model = PolicyNetwork::<Backend>::default();
        let mut trainer = PolicyTrainer::with_config(model, AdamConfig::new(), 1.0e-3);
        let game = GameBuilder::new(2).expect("builder").build().expect("game");
        let view = game.state_view(0).expect("state view");
        let states = StateEncoder::encode_tensor::<Backend>(&view);
        let targets =
            ActionSpace::target_tensor::<Backend>(&[
                ActionSpace::action_index(&Action::EndTurn).unwrap()
            ]);
        let masks = ActionSpace::mask_tensor::<Backend>(&[Action::EndTurn]);
        let batch = PolicyBatch::new(states, targets, masks);
        let loss = trainer.train_step(batch);
        assert!(loss.is_finite());
    }

    #[test]
    fn dataset_split_respects_fraction() {
        let sample = PolicySample::from_components(
            [0.0; STATE_FEATURES],
            ActionSpace::mask(&[Action::EndTurn]),
            ActionSpace::targets_from_indices(&[
                ActionSpace::action_index(&Action::EndTurn).unwrap()
            ]),
            1.0,
        );
        let mut dataset = PolicyDataset::new();
        dataset.extend(std::iter::repeat(sample).take(20));
        let mut rng = StdRng::seed_from_u64(42);
        let (train, validation) = dataset.split(0.2, &mut rng);
        assert!(train.len() > 0);
        assert!(validation.len() > 0);
        assert_eq!(train.len() + validation.len(), 20);
    }

    #[test]
    fn policy_batch_from_samples_preserves_shapes() {
        let sample = PolicySample::from_components(
            [0.1; STATE_FEATURES],
            ActionSpace::mask(&[Action::EndTurn]),
            ActionSpace::targets_from_indices(&[
                ActionSpace::action_index(&Action::EndTurn).unwrap()
            ]),
            2.5,
        );
        let batch = PolicyBatch::<Backend>::from_samples(&[sample.clone(), sample]);
        assert_eq!(batch.sample_count(), 2);
        assert!(batch.weight_sum() > 0.0);
        assert_eq!(batch.states.shape().dims, [2, STATE_FEATURES]);
        assert_eq!(batch.targets.shape().dims, [2, ActionSpace::MAX]);
    }

    #[test]
    fn trainer_fit_returns_metrics() {
        let mut dataset = PolicyDataset::new();
        let sample = PolicySample::from_components(
            [0.0; STATE_FEATURES],
            ActionSpace::mask(&[Action::EndTurn]),
            ActionSpace::targets_from_indices(&[
                ActionSpace::action_index(&Action::EndTurn).unwrap()
            ]),
            1.0,
        );
        dataset.extend(std::iter::repeat(sample).take(32));
        let mut rng = StdRng::seed_from_u64(7);
        let (mut train, validation) = dataset.split(0.25, &mut rng);
        let mut trainer = PolicyTrainer::with_config(
            PolicyNetwork::<Backend>::default(),
            AdamConfig::new(),
            1.0e-3,
        );
        let history = trainer.fit(
            &mut train,
            Some(&validation),
            TrainingLoopConfig {
                epochs: 3,
                batch_size: 8,
            },
            &mut rng,
        );
        assert_eq!(history.len(), 3);
        assert!(history.iter().all(|metrics| metrics.samples > 0));
    }
}
