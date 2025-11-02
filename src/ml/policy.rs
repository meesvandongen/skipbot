use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation::relu;
use burn::tensor::backend::Backend;

use crate::state::GameStateView;

use super::encoding::{ActionSpace, STATE_FEATURES, StateEncoder};

pub const DEFAULT_HIDDEN: usize = 128;
pub const DEFAULT_STACK: usize = 2;
pub const DEFAULT_OUTPUT: usize = ActionSpace::MAX;

#[derive(Module, Debug)]
pub struct PolicyNetwork<B: Backend> {
    stack: Vec<Linear<B>>,
    output: Linear<B>,
}

impl<B> PolicyNetwork<B>
where
    B: Backend,
    B::Device: Default,
{
    pub fn new(hidden: usize, stack_depth: usize) -> Self {
        assert!(stack_depth > 0, "stack depth must be positive");
        let mut stack = Vec::with_capacity(stack_depth);
        let device = B::Device::default();
        let mut input_size = STATE_FEATURES;
        for _ in 0..stack_depth {
            let layer = LinearConfig::new(input_size, hidden).init(&device);
            stack.push(layer);
            input_size = hidden;
        }
        let output = LinearConfig::new(input_size, ActionSpace::MAX).init(&device);
        Self { stack, output }
    }

    pub fn default() -> Self {
        Self::new(DEFAULT_HIDDEN, DEFAULT_STACK)
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut activations = input;
        for layer in &self.stack {
            activations = layer.forward(activations);
            activations = relu(activations);
        }
        self.output.forward(activations)
    }

    pub fn forward_state(&self, state: &GameStateView) -> Tensor<B, 1> {
        let batch = StateEncoder::encode_tensor::<B>(state);
        self.forward(batch).reshape([ActionSpace::MAX])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::GameBuilder;
    use burn_ndarray::NdArray;

    #[test]
    fn forward_produces_expected_shape() {
        let network = PolicyNetwork::<NdArray<f32>>::default();
        let game = GameBuilder::new(2).expect("builder").build().expect("game");
        let view = game.state_view(0).expect("state view");
        let batch = StateEncoder::encode_tensor::<NdArray<f32>>(&view);
        let logits = network.forward(batch);
        let shape = logits.shape();
        assert_eq!(shape.dims[0], 1);
        assert_eq!(shape.dims[1], ActionSpace::MAX);
    }
}
