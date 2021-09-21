use juice::layer::{LayerConfig, LayerType};
use juice::layers::{
    ConvolutionConfig, LinearConfig, PoolingConfig, PoolingMode, SequentialConfig,
};

use image::io::Reader as ImageReader;

pub struct AlexNet {
    batch_size: usize,
    input_width: usize,
    input_height: usize,
    network: SequentialConfig,
}

#[cfg(all(feature = "cuda"))]
impl AlexNet {
    pub fn new(input_width: usize, input_height: usize, options: Option<usize>) -> AlexNet {
        let batch_size;
        if let Some(x) = options {
            batch_size = x;
        } else {
            batch_size = 16;
        }

        let mut network = SequentialConfig::default();

        // output: 224 x 224 x 3
        network.add_input("data", &[batch_size, 3, input_height, input_width]);

        // output: 54 x 54 x 96
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 96,
                filter_shape: vec![11],
                padding: vec![0],
                stride: vec![4],
            },
        ));
        network.add_layer(LayerConfig::new("relu", LayerType::ReLU));

        // output: 26 x 26 x 96
        network.add_layer(LayerConfig::new(
            "pool",
            PoolingConfig {
                mode: PoolingMode::Max,
                filter_shape: vec![3],
                padding: vec![0],
                stride: vec![2],
            },
        ));

        // output: 26 x 26 x 256
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 256,
                filter_shape: vec![5],
                padding: vec![2],
                stride: vec![1],
            },
        ));
        network.add_layer(LayerConfig::new("relu", LayerType::ReLU));

        // output: 12 x 12 x 256
        network.add_layer(LayerConfig::new(
            "pool",
            PoolingConfig {
                mode: PoolingMode::Max,
                filter_shape: vec![3],
                padding: vec![0],
                stride: vec![2],
            },
        ));

        // output: 12 x 12 x 384
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 384,
                filter_shape: vec![3],
                padding: vec![1],
                stride: vec![1],
            },
        ));
        network.add_layer(LayerConfig::new("relu", LayerType::ReLU));

        // output: 12 x 12 x 384
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 384,
                filter_shape: vec![3],
                padding: vec![1],
                stride: vec![1],
            },
        ));
        network.add_layer(LayerConfig::new("relu", LayerType::ReLU));

        // output: 12 x 12 x 256
        network.add_layer(LayerConfig::new(
            "conv",
            ConvolutionConfig {
                num_output: 256,
                filter_shape: vec![3],
                padding: vec![1],
                stride: vec![1],
            },
        ));
        network.add_layer(LayerConfig::new("relu", LayerType::ReLU));

        // output: 5 x 5 x 256
        network.add_layer(LayerConfig::new(
            "pool",
            PoolingConfig {
                mode: PoolingMode::Max,
                filter_shape: vec![3],
                padding: vec![0],
                stride: vec![2],
            },
        ));

        network.add_layer(LayerConfig::new("fully-connected", LinearConfig { output_size: 4096 }));
        network.add_layer(LayerConfig::new("fully-connected", LinearConfig { output_size: 4096 }));
        network.add_layer(LayerConfig::new("fully-connected", LinearConfig { output_size: 1000 }));

        AlexNet {
            batch_size: batch_size,
            input_width: input_width,
            input_height: input_height,
            network: network.clone(),
        }
    }

    pub fn train(directory: String, batch_size: u16) {
        // TODO
    }
}
