#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using DataLoaderType = torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>;

void createDirectory();
auto transform();
auto createDataloader();
void checkDataSizePerBatch(DataLoaderType& dataloader) {
    std::cout << "-- Checking Data Size Per Batch --" << endl;
    // Iterate through the DataLoader
    for (auto& batch : dataloader) {
        auto data = batch.data;
        auto targets = batch.target;
        std::cout << "Batch data size: " << data.sizes() << std::endl;
        std::cout << "Batch targets size: " << targets.sizes() << std::endl;
        std::cout << "\n" << endl;
        break;
    }
};
