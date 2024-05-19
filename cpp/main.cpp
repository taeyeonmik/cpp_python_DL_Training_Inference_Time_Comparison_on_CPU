#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

#include "lib/ConvNetClassifier.hpp"
#include "lib/Dataset.hpp"
#include "lib/utils.hpp"

int main() {
    // File path to the MNIST images
    const std::string base_path = "/Users/taeyeon/PersonalProjects/cpp_python_DeepLearning_Comparison_Inference_Time/infer-py/MNIST/raw/";
    const std::string train_image_path = base_path + "train-images-idx3-ubyte";
    const std::string train_label_path = base_path + "train-labels-idx1-ubyte";
    const std::string test_image_path = base_path + "t10k-images-idx3-ubyte";
    const std::string test_label_path = base_path + "t10k-labels-idx1-ubyte";
    // MNIST TRAINSET
    Mnist MnistTrain(train_image_path, train_label_path);
    torch::Tensor trainImgs = MnistTrain.ImgTensor;
    torch::Tensor trainLabels = MnistTrain.LabelTensor;
    // MNIST TESTSET
    Mnist MnistTest(train_image_path, train_label_path);
    torch::Tensor testImgs = MnistTest.ImgTensor;
    torch::Tensor testLabels = MnistTest.LabelTensor;

    // create torch custom datasets
    trainImgs = trainImgs.to(torch::kFloat32) / 255.0;
    auto train_dataset = CustomDataset(trainImgs, trainLabels).map(torch::data::transforms::Stack<>()); //Stack transforms batches into a single tensor
    testImgs = testImgs.to(torch::kFloat32) / 255.0;
    auto test_dataset = CustomDataset(testImgs, testLabels).map(torch::data::transforms::Stack<>()); //Stack transforms batches into a single tensor

    // create dataloaders
    int bs = 32;
    auto train_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(train_dataset), bs);
    auto test_dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(test_dataset), bs);

    // check data size for a batch
    checkDataSizePerBatch(*train_dataloader);

    return 0;
}