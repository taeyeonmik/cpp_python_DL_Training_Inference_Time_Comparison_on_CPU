#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>
#include <tuple>
#include <torch/script.h>
#include <torch/torch.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using DataLoaderType = torch::data::StatelessDataLoader<torch::data::datasets::MapDataset<CustomDataset, torch::data::transforms::Stack<>>, torch::data::samplers::RandomSampler>;

void checkDataSizePerBatch(DataLoaderType& dataloader) {
    std::cout << "-- Checking Data Size Per Batch --" << endl;
    // Iterate through the DataLoader
    for (torch::data::Example<>& batch : dataloader) {
        auto data = batch.data;
        auto targets = batch.target;
        std::cout << "Batch data size: " << data.sizes() << std::endl;
        std::cout << "Batch targets size: " << targets.sizes() << std::endl;
        std::cout << "\n" << endl;
        break;
    }
}
// get output size
std::tuple<uint8_t, uint8_t> getOutputSize(uint8_t number_conv, uint8_t k=3, uint8_t p=1,
                                           uint8_t s=1, uint8_t poolk=2, uint8_t pools=1) {
    uint8_t win = 28;
    uint8_t hin = 28;
    uint8_t wout, hout;
    for (uint8_t conv=0; conv<number_conv; ++conv) {
        // convolution
        wout = (win - k + 2 * p) / s + 1;
        hout = (hin - k + 2 * p) / s + 1;
        // pooling
        wout = (wout - poolk) / pools + 1;
        hout = (hout - poolk) / pools + 1;
        // init
        win = wout;
        hin = hout;
    }
    return std::make_tuple(wout, hout);
}
//torch::Device checkDevice() {
//    // Set device
//    torch::Device device(torch::kCPU);
//    if (torch::cuda::is_available()) {
//        device = torch::Device(torch::kCUDA);
//    }
//    return device;
//}
// Read model (from python scripted model)
//torch::jit::script::Module read_model(std::string model_path, bool usegpu)
//{
//
//    torch::jit::script::Module model = torch::jit::load(model_path);
//
//    if (usegpu)
//    {
//        torch::DeviceType gpu_device_type = torch::kCUDA;
//        torch::Device gpu_device(gpu_device_type);
//
//        model.to(gpu_device);
//    }
//    else
//    {
//        torch::DeviceType cpu_device_type = torch::kCPU;
//        torch::Device cpu_device(cpu_device_type);
//
//        model.to(cpu_device);
//    }
//
//    return model;
//}

#endif