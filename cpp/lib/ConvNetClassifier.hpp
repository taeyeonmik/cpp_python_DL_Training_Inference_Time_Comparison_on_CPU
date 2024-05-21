#ifndef CONVNETCLASSIFIER_HPP
#define CONVNETCLASSIFIER_HPP

#include <iostream>
#include <vector>
#include <chrono>
#include <torch/script.h>
#include <torch/torch.h>

#include "ConvNet.hpp"
#include "Dataset.hpp"
#include "utils.hpp"

using DataLoaderType = torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>;

class ConvNetClassifier {
public:
    torch::Device *device;
    ConvNet *model;
    torch::optim::Adam *optimizer;
    std::vector<double> train_loss;
    std::vector<double> valid_loss;
    std::vector<double> valid_accuracy;

    // constructor
    ConvNetClassifier(uint16_t numClasses, double learningRate) {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        cout << "Device: " << *device << endl;
        cout << "ConvNet Classifier is being compiled...\nnumClasses: " << numClasses << "\n" << "learningRate: " << learningRate << "\n" << "optimizer: Adam" << endl;
        std::tuple<uint8_t, uint8_t> wh_out = getOutputSize(3);
        model = new ConvNet(numClasses, std::get<0>(wh_out), std::get<1>(wh_out));
        (*model).to(*device);
        optimizer = new torch::optim::Adam(
                (*model).parameters(),
                torch::optim::AdamOptions(learningRate)
        );
    }
    // constructor for inference
    ConvNetClassifier(uint16_t numClasses, std::string modelPath) {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        cout << "Device: " << *device << endl;
        std::tuple<uint8_t, uint8_t> wh_out = getOutputSize(3);
        model = new ConvNet(numClasses, std::get<0>(wh_out), std::get<1>(wh_out));

        // Load model state
        torch::serialize::InputArchive input_archive;
        input_archive.load_from(modelPath);
        model->load(input_archive);

        cout << "ConvNet Classifier is compiled with \"model.pt\"" << endl;
        (*model).to(*device);
    }
    void train(DataLoaderType& train_dataloader, DataLoaderType& valid_dataloader, size_t num_epochs, uint16_t train_step) {
        auto criterion = torch::nn::CrossEntropyLoss();
        for (size_t epoch=0; epoch<num_epochs; ++epoch) {
            size_t batch_idx = 0;
            double total_loss = 0.0f;
            auto start = std::chrono::high_resolution_clock::now();
            for (torch::data::Example<>& batch : train_dataloader) {
                auto data = batch.data.to(*device);
                auto targets = batch.target.to(*device);

                optimizer->zero_grad();
                torch::Tensor output = model->forward(data);
                torch::Tensor loss = criterion(output, targets);

                loss.backward();
                optimizer->step();

                total_loss += loss.item<double>();
                // Print loss
                if (batch_idx++ % 10 == 0) {
                    std::cout << "Train Epoch: ["<< epoch + 1 << "]\tStep: [" << batch_idx << "/" << train_step << "]\tLoss: " << loss.item<double>() << std::endl;
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            double average_loss = total_loss / (double) batch_idx;
            train_loss.push_back(total_loss);

            // validation
            size_t batch_val_idx = 0;
            double total_val_loss = 0.0f;
            double total_accuracy = 0.0f;
            for (torch::data::Example<>& batch : valid_dataloader) {
                auto data = batch.data.to(*device);
                auto targets = batch.target.to(*device);
                {
                    torch::NoGradGuard no_grad;
                    auto [val_loss, accuracy] = evaluate(*model, data, targets, criterion);
                    total_val_loss += val_loss;
                    total_accuracy += accuracy;
                }
                batch_val_idx += 1;
            }
            double average_val_loss = total_val_loss / (double) batch_val_idx;
            double average_val_accuracy = total_accuracy / (double) batch_val_idx;
            valid_loss.push_back(total_val_loss);
            valid_accuracy.push_back(total_accuracy);
            // Print total loss, average loss, average accuracy and time taken
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Total Train Loss: " << total_loss << ", Average Train Loss: " << average_loss << ", Time: " << elapsed.count() << " seconds" << std::endl;
            std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "], Total Valid Loss: " << total_val_loss << ", Average Valid Loss: " << average_val_loss << ", Average Valid Accuracy: " << average_val_accuracy << endl;

            // save model
            std::string model_path = "../model.pt";
            torch::serialize::OutputArchive output_archive;
            model->save(output_archive);
            output_archive.save_to(model_path);
        }
    }
    // Dummy function for validation (replace with your actual validation function)
    std::pair<double, double> evaluate(ConvNet& model, torch::Tensor input, torch::Tensor target, torch::nn::CrossEntropyLoss criterion) {
        torch::Tensor output = model.forward(input);
        torch::Tensor loss = criterion(output, target);
        // Calculate accuracy
        torch::Tensor pred = torch::argmax(output, 1);
        double accuracy = (pred == target).sum().item<double>() / target.size(0);

        return {loss.item<double>(), accuracy};
    }
};

#endif