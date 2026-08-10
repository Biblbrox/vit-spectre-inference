
use burn::{
    Tensor,
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig},
    tensor::{Device, Shape},
};

use crate::{
    attention::dsthaattention::{DSTHA, DSTHAConfig},
    augmentations::DropPath,
    norm::{DynamicERF, DynamicERFConfig},
};

#[derive(Module, Debug)]
pub struct TokenMerger {
    pos: Param<Tensor<3>>,
    proj: Linear,
    scale: Param<Tensor<3>>,
    
}

#[derive(Config, Debug)]
pub struct TokenMergerConfig {
    embed_dim: usize,
    seq_length: usize,
}

impl TokenMergerConfig {
    pub fn init(&self, device: &Device) -> TokenMerger {
        let out_seq = (self.seq_length / 2).max(1);

        TokenMerger {
            pos: Param::<Tensor<3>>::from_tensor(Tensor::<3>::zeros(
                Shape::new([1, out_seq, self.embed_dim]),
                device,
            ))
            .set_require_grad(true),
            proj: LinearConfig::new(self.embed_dim * 2, self.embed_dim)
                .with_bias(false)
                .init(device),
            scale: Param::from_tensor(Tensor::<3>::zeros([1, 1, 1], device))
                .set_require_grad(true),
        }
    }
}

impl TokenMerger {
    /// x: [B, N, E] → [B, N/2, E]
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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
pub struct FastEncoderLayer {
    linear1: Linear,
    linear2: Linear,
    //mix_layer: StochasticWindowMixer,
    //mix_layer: StochasticAttention,
    //mix_layer: StochasticAttentionStaticWindow,
    mix_layer: DSTHA,
    norm1: DynamicERF,
    norm2: DynamicERF,
    dropout: Dropout,
    activation: Gelu,
    drop_path: DropPath,

    merger: TokenMerger,
    
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
pub struct FastEncoder {
    encoder_layers: Vec<FastEncoderLayer>,
    norm: Option<DynamicERF>,
    
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

impl FastEncoderLayer {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        // Mixing block with stochastic depth
        let mix_out = self.mix_layer.forward(self.norm1.forward(x.clone()));
        let mix_out = self.dropout.forward(mix_out);
        let x = self.drop_path.forward(x, mix_out);

        // FFN block with stochastic depth
        let ff_out = self._ff_block(self.norm2.forward(x.clone()));

        self.drop_path.forward(x, ff_out)
    }

    pub fn _ff_block(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = self.linear1.forward(x);
        self.dropout.forward(
            self.linear2
                .forward(self.dropout.forward(self.activation.forward(hidden))),
        )
    }
}

impl FastEncoderLayerConfig {
    pub fn init(&self, device: &Device) -> FastEncoderLayer {
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

impl FastEncoder {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
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
    pub fn init(&self, device: &Device) -> FastEncoder {
        let mut layers = Vec::new();

        for _i in 0..self.num_layers {
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
