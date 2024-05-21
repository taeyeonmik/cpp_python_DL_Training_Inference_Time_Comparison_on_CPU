#include <iostream>
#include <vector>
#include <cmath>
#include <torch/torch.h>
#include <torch/script.h>

#include "lib/ConvNetClassifier.hpp"
#include "lib/Dataset.hpp"
#include "lib/utils.hpp"

int main() {
    // File path to the MNIST images
    const std::string base_path = "/Users/taeyeon/PersonalProjects/cpp_python_DL_Training_Inference_Time_Comparison_on_CPU/python/MNIST/raw/";
    const std::string train_image_path = base_path + "train-images-idx3-ubyte";
    const std::string train_label_path = base_path + "train-labels-idx1-ubyte";
    const std::string test_image_path = base_path + "t10k-images-idx3-ubyte";
    const std::string test_label_path = base_path + "t10k-labels-idx1-ubyte";

    Mnist MnistTrain(train_image_path, train_label_path);
    torch::Tensor Imgs = MnistTrain.ImgTensor;
    torch::Tensor Labels = MnistTrain.LabelTensor;
    // Split
    auto splitImgsTensor = torch::split(Imgs, {48000, 12000}, 0);
    auto splitLabelsTensor = torch::split(Labels, {48000, 12000}, 0);
    // MNIST TRAINSET
    torch::Tensor trainImgs = splitImgsTensor[0];
    torch::Tensor trainLabels = splitLabelsTensor[0];
    // MNIST VALIDSET
    torch::Tensor validImgs = splitImgsTensor[1];
    torch::Tensor validLabels = splitLabelsTensor[1];
    // MNIST TESTSET
    Mnist MnistTest(test_image_path, test_label_path);
    torch::Tensor testImgs = MnistTest.ImgTensor;
    torch::Tensor testLabels = MnistTest.LabelTensor;
    std::cout << "Train dataset shape: " << trainImgs.sizes() << std::endl;
    std::cout << "Validation dataset shape: " << validImgs.sizes() << std::endl;
    std::cout << "Test dataset shape: " << testImgs.sizes() << std::endl;

    // create torch custom datasets
    trainImgs = trainImgs.to(torch::kFloat32) / 255.0;
    auto train_dataset = CustomDataset(trainImgs, trainLabels).map(torch::data::transforms::Stack<>()); //Stack transforms batches into a single tensor
    validImgs = validImgs.to(torch::kFloat32) / 255.0;
    auto valid_dataset = CustomDataset(validImgs, validLabels).map(torch::data::transforms::Stack<>());
    testImgs = testImgs.to(torch::kFloat32) / 255.0;
    auto test_dataset = CustomDataset(testImgs, testLabels).map(torch::data::transforms::Stack<>());

    uint16_t train_datasize = trainImgs.size(0);
    uint16_t valid_datasize = validImgs.size(0);
    uint16_t test_datasize = testImgs.size(0);

    // create dataloaders
    uint16_t bs = 64;
    uint16_t train_step = std::ceil(train_datasize / bs);
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), bs);
    auto valid_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(valid_dataset), bs);
    auto test_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(test_dataset), bs);

    // check data size for a batch
    checkDataSizePerBatch(*train_dataloader);

    // configs
    uint8_t num_epochs = 10;
    uint16_t numClasses = 10;
    double learningRate = 0.001;
    // model
    ConvNetClassifier Net(numClasses, learningRate);

    // train
    Net.train(*train_dataloader, *valid_dataloader, num_epochs, train_step);

    // Inference
//    std::string modelPath = "../model.pt";
//    ConvNetClassifier Net(numClasses, modelPath);
//
//    Net->evaluate() //
//    return 0;
}