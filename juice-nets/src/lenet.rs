use juice::layers::{SequentialConfig, ConvolutionConfig};
use juice::layer::LayerConfig;

pub struct LeNet {
    batch_size: usize,
    input_width: usize,
    input_height: usize,
    network: SequentialConfig,
}

impl LeNet {
    #[cfg(all(feature="cuda"))]
    pub fn new(input_width: usize, input_height: usize, options: Option<usize>) -> LeNet {
        let batch_size;
        if let Some(x) = options {
            batch_size = x;
        } else {
            batch_size = 16;
        }

        let mut network = SequentialConfig::default();
        network.add_input("data", &[batch_size, input_width, input_width]);

        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 6,
                filter_shape: vec![3],
                padding: vec![0],
                stride: vec![1],
            },
        ));

        LeNet {
            batch_size: batch_size,
            input_width: input_width,
            input_height: input_height,
            network: network.clone()
        }
    }
}