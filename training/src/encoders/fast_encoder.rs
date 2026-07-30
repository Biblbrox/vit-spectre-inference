use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    tensor::{Shape, backend::Backend},
};

use crate::{
    attention::{
        dsthaattention::{DSTHA, DSTHAConfig},
        stochasticmixer::{StochasticAttention, StochasticAttentionConfig},
        stochasticmixerstatic::{StochasticAttentionStatic, StochasticAttentionStaticConfig},
        stochasticmixerstaticwindow::{
            StochasticAttentionStaticWindow, StochasticAttentionStaticWindowConfig,
        },
    },
    augmentations::DropPath,
    norm::{DynamicERF, DynamicERFConfig},
};

#[derive(Module, Debug)]
pub struct TokenMerger<B: Backend> {
    pos: Param<Tensor<B, 3>>,
    proj: Linear<B>,
    scale: Param<Tensor<B, 3>>,
}

#[derive(Config, Debug)]
pub struct TokenMergerConfig {
    embed_dim: usize,
    seq_length: usize,
}

impl TokenMergerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TokenMerger<B> {
        let out_seq = (self.seq_length / 2).max(1);

        TokenMerger {
            pos: Param::<Tensor<B, 3>>::from_tensor(Tensor::<B, 3>::zeros(
                Shape::new([1, out_seq, self.embed_dim]),
                device,
            ))
            .set_require_grad(true),
            proj: LinearConfig::new(self.embed_dim * 2, self.embed_dim)
                .with_bias(false)
                .init(device),
            scale: Param::from_tensor(Tensor::<B, 3>::zeros([1, 1, 1], device))
                .set_require_grad(true),
        }
    }
}

impl<B: Backend> TokenMerger<B> {
    /// x: [B, N, E] → [B, N/2, E]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, n, e] = x.dims();

        if n <= 1 {
            return x;
        }

        let half = n / 2;

        let dst = x.clone().slice([0..b, 0..half, 0..e]); // [B, N/2, E]
        let src = x.slice([0..b, half..n, 0..e]); // [B, N/2, E]

        self.proj
            .forward(Tensor::cat(vec![src.clone(), dst.clone()], 2))
            * self.scale.val()
            + (dst + src) / 2.0
            + self.pos.val()
    }
}

#[derive(Module, Debug)]
pub struct FastEncoderLayer<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    //mix_layer: StochasticWindowMixer<B>,
    //mix_layer: StochasticAttention<B>,
    //mix_layer: StochasticAttentionStaticWindow<B>,
    mix_layer: DSTHA<B>,
    norm1: DynamicERF<B>,
    norm2: DynamicERF<B>,
    dropout: Dropout,
    activation: Gelu,
    drop_path: DropPath<B>,

    merger: TokenMerger<B>,
}

#[derive(Config, Debug)]
pub struct FastEncoderLayerConfig {
    seq_length: usize,
    embed_dim: usize,
    hidden_dim: usize,
    dropout: f64,
    #[config(default = 0.0)]
    drop_path_prob: f64,
    nhead: usize,
}

#[derive(Module, Debug)]
pub struct FastEncoder<B: Backend> {
    encoder_layers: Vec<FastEncoderLayer<B>>,
    norm: Option<DynamicERF<B>>,
}

#[derive(Config, Debug)]
pub struct FastEncoderConfig {
    num_layers: usize,
    seq_length: usize,
    embed_dim: usize,
    hid_dim: usize,
    dropout: f64,
    nhead: usize,
}

impl<B: Backend> FastEncoderLayer<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        // Mixing block with stochastic depth
        let mix_out = self.mix_layer.forward(self.norm1.forward(x.clone()));
        let mix_out = self.dropout.forward(mix_out);
        let x = self.drop_path.forward(x, mix_out);

        // FFN block with stochastic depth
        let ff_out = self._ff_block(self.norm2.forward(x.clone()));

        self.drop_path.forward(x, ff_out)
    }

    pub fn _ff_block(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.linear1.forward(x);
        self.dropout.forward(
            self.linear2
                .forward(self.dropout.forward(self.activation.forward(hidden))),
        )
    }
}

impl FastEncoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastEncoderLayer<B> {
        FastEncoderLayer {
            linear1: LinearConfig::new(self.embed_dim, self.hidden_dim).init(device),
            linear2: LinearConfig::new(self.hidden_dim, self.embed_dim).init(device),
            //mix_layer: StochasticWindowMixerConfig::new(
            //    self.embed_dim,
            //    self.seq_length,
            //    self.nhead,
            //    3,
            //    0.05,
            //)
            //.init(device),
            //mix_layer: StochasticAttentionConfig::new(self.embed_dim, self.nhead, 0.05)
            //    .init(device),
            //mix_layer: StochasticAttentionStaticWindowConfig::new(
            //    self.embed_dim,
            //    self.seq_length,
            //    self.nhead,
            //    3,
            //    0.05,
            //)
            //.init(device),
            mix_layer: DSTHAConfig::new(self.embed_dim, self.nhead).init(device),
            norm1: DynamicERFConfig::new(self.embed_dim).init(device),
            norm2: DynamicERFConfig::new(self.embed_dim).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            drop_path: DropPath::new(self.drop_path_prob),
            activation: Gelu::new(),
            merger: TokenMergerConfig::new(self.embed_dim, self.seq_length).init(device),
        }
    }
}

impl<B: Backend> FastEncoder<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut output = x.clone();

        for layer in self.encoder_layers.iter() {
            output = layer.forward(output);
        }

        if let Some(norm) = &self.norm {
            output = norm.forward(output);
        }

        output
    }
}

impl FastEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FastEncoder<B> {
        let mut layers = Vec::new();

        for i in 0..self.num_layers {
            layers.push(
                FastEncoderLayerConfig::new(
                    self.seq_length,
                    self.embed_dim,
                    self.hid_dim,
                    self.dropout,
                    self.nhead,
                )
                //.with_drop_path_prob(((i + 1) as f64 / self.num_layers as f64) * 0.1)
                .init(device),
            );
        }
        FastEncoder {
            encoder_layers: layers,
            norm: Option::None,
        }
    }
}
