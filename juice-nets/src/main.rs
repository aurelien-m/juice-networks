mod alexnet;
mod lenet;

fn main() {
    // let lenet_newtork = lenet::LeNet::new(0, 0, Option::None);
    let alexnet = alexnet::AlexNet::new(227, 227, None);

    alexnet.train(
        "/home/aurelien/Documents/Datasets/dog-breed-identification/labels.csv",
        "/home/aurelien/Documents/Datasets/dog-breed-identification/train/",
        16,
    );
}
