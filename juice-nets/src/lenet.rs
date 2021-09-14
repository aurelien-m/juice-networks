use juice::layers::{SequentialConfig, ConvolutionConfig, PoolingConfig, PoolingMode, LinearConfig};
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

        // output: 32 x 32 x 1
        network.add_input("data", &[batch_size, input_width, input_width]);

        // output: 28 x 28 x 6
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 6,
                filter_shape: vec![5],
                padding: vec![0],
                stride: vec![1],
            },
        ));

        // output: 14 x 14 x 6
        network.add_layer(LayerConfig::new(
            "pooling",
            PoolingConfig {
                mode: PoolingMode::Max,
                filter_shape: vec![2],
                padding: vec![0],
                stride: vec![2],
            },
        ));

        // output: 10 x 10 x 16
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 16,
                filter_shape: vec![5],
                padding: vec![0],
                stride: vec![1],
            },
        ));

        // output: 5 x 5 x 16
        network.add_layer(LayerConfig::new(
            "pooling",
            PoolingConfig {
                mode: PoolingMode::Max,
                filter_shape: vec![2],
                padding: vec![0],
                stride: vec![2],
            },
        ));

        // output: 120
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 120,
                filter_shape: vec![5],
                padding: vec![0],
                stride: vec![1],
            },
        ));

        // output: 84
        network.add_layer(LayerConfig::new(
            "fully-connected-layer",
            LinearConfig { output_size: 84 },
        ));

        // 1output: 10
        network.add_layer(LayerConfig::new(
            "output",
            LinearConfig { output_size: 10 },
        ));

        LeNet {
            batch_size: batch_size,
            input_width: input_width,
            input_height: input_height,
            network: network.clone()
        }
    }
}