use burn::{config::Config, module::Module, prelude::Tensor, tensor::backend::Backend};

use crate::attention::{
    NormalizationMode,
    self_attention::{SelfAttention, SelfAttentionConfig},
    sinkhorn,
};

/// Sinkformer implementation according to https://arxiv.org/pdf/2110.11773

#[derive(Config, Debug)]
pub struct SinkformerMixerConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub kernel_size: usize,
    pub temperature: f32,
}

#[derive(Module, Debug)]
pub struct SinkformerMixer<B: Backend> {
    mha_attn: SelfAttention<B>,
    temperature: f32,
    n_heads: usize,
}

impl<B: Backend> SinkformerMixer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let (scores, v) = self.mha_attn.scores(x);
        let scores = sinkhorn(scores, self.temperature, NormalizationMode::Double);
        self.mha_attn.apply_output(scores, v)
    }
}

impl SinkformerMixerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SinkformerMixer<B> {
        SinkformerMixer {
            temperature: self.temperature,
            n_heads: self.n_heads,
            mha_attn: SelfAttentionConfig::new(self.d_model, self.n_heads).init(device),
        }
    }
}
