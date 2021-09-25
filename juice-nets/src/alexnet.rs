use std::fs::read_dir;
use std::io::Result;
use std::path::Path;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use coaster::frameworks::cuda::get_cuda_backend;
use coaster::SharedTensor;
use juice::layer::{LayerConfig, LayerType};
use juice::layers::{
    ConvolutionConfig, LinearConfig, PoolingConfig, PoolingMode, SequentialConfig,
};

use csv::{self, StringRecord};
use serde::Deserialize;

pub struct AlexNet {
    batch_size: usize,
    input_width: usize,
    input_height: usize,
    network: SequentialConfig,
}

#[derive(Deserialize)]
struct Row {
    id: String,
    breed: String,
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

        network.add_layer(LayerConfig::new(
            "fully-connected",
            LinearConfig { output_size: 4096 },
        ));
        network.add_layer(LayerConfig::new(
            "fully-connected",
            LinearConfig { output_size: 4096 },
        ));
        network.add_layer(LayerConfig::new(
            "fully-connected",
            LinearConfig { output_size: 1000 },
        ));

        AlexNet {
            batch_size: batch_size,
            input_width: input_width,
            input_height: input_height,
            network: network.clone(),
        }
    }

    pub fn train(&self, csv_map: &str, train_directory: &str, batch_size: usize) -> Result<()> {
        let backend = Rc::new(get_cuda_backend());

        let input = SharedTensor::<f32>::new(&[
            batch_size,
            3 as usize,
            self.input_height,
            self.input_width,
        ]);
        let input_lock = Arc::new(RwLock::new(input));

        let label = SharedTensor::<f32>::new(&[batch_size as usize, 1]);
        let label_lock = Arc::new(RwLock::new(label));

        let dataset_path = Path::new(&train_directory);
        let dataset_directory = read_dir(dataset_path).unwrap();

        let header = StringRecord::from(vec!["id", "breed"]);

        let csv_reader = csv::Reader::from_path(&csv_map);
        let mut train_images = csv_reader.unwrap();

        for batch_n in 0..(dataset_directory.count() / batch_size as usize) {
            let mut input_tensor = input_lock.write().unwrap();
            let mut label_tensor = label_lock.write().unwrap();

            for image_n in batch_n..(batch_n + batch_size) {
                match train_images.records().next().unwrap() {
                    Ok(record) => {
                        let row: Row = record.deserialize(Some(&header))?;
                        println!("id: {:?} - breed: {:?}", row.id, row.breed);
                    }
                    Err(_) => {
                        // TODO ...
                    }
                }
            }

            // let image_bytes =
        }

        Ok(())
    }
}
