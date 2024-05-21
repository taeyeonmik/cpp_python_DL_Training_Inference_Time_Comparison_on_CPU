#ifndef DATASET_HPP
#define DATASET_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <cstdint>

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    torch::Tensor images_;
    torch::Tensor labels_;
public:
    CustomDataset(torch::Tensor images, torch::Tensor labels)
            : images_(images), labels_(labels) {}

    // Override the get method to return a tensor for the given index
    torch::data::Example<> get(size_t index) override {
        return {images_[index], labels_[index]};
    }
    // Method to return a tensor for the given index directly
    torch::Tensor get_image(size_t index) const {
        return images_[index];
    }

    // Method to return a label for the given index directly
    torch::Tensor get_label(size_t index) const {
        return labels_[index];
    }
    // Return the size of the dataset
    torch::optional<size_t> size() const override {
        return images_.size(0);
    }
};

class Mnist {
public:
    int32_t num_images, num_rows, num_cols;
    int32_t num_labels;
    torch::Tensor ImgTensor;
    torch::Tensor LabelTensor;

    Mnist(const std::string& ImgPath, const std::string& LabelPath) {
        // read elements
        std::vector<uint8_t> Img = read_mnist_element(ImgPath, num_images, num_rows, num_cols);
        std::vector<uint8_t> Label = read_mnist_element(LabelPath, num_labels);
        // convert the elements to tensor
        ImgTensor = convert_to_tensor(Img, num_images, num_rows, num_cols);
        LabelTensor = convert_to_tensor(Label, num_labels);
    };
    std::ifstream readFile(const std::string& filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }

        int32_t magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number); // Byte swap to handle endianness

        if (magic_number != 2051) { // image
            throw std::runtime_error("Invalid MNIST image/label file!");
        }
        return file;
    }
    std::ifstream readLabelFile(const std::string& filePath) {
        std::ifstream file(filePath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filePath);
        }

        int32_t magic_number = 0;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = __builtin_bswap32(magic_number); // Byte swap to handle endianness

        if (magic_number != 2049) { // label
            throw std::runtime_error("Invalid MNIST image/label file!");
        }
        return file;
    }
    // Function to read MNIST image file
    std::vector<uint8_t> read_mnist_element(const std::string& file_path, int32_t& num_elements, int32_t& num_rows, int32_t& num_cols) {
        std::ifstream file = readFile(file_path);
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        num_elements = __builtin_bswap32(num_elements);
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        num_rows = __builtin_bswap32(num_rows);
        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        num_cols = __builtin_bswap32(num_cols);

        std::vector<uint8_t> images(num_elements * num_rows * num_cols);
        file.read(reinterpret_cast<char*>(images.data()), images.size());
        file.close();
        return images;
    }
    // Function to read MNIST label file
    std::vector<uint8_t> read_mnist_element(const std::string& file_path, int32_t& num_elements) {
        std::ifstream file = readLabelFile(file_path);
        file.read(reinterpret_cast<char*>(&num_elements), sizeof(num_elements));
        num_elements = __builtin_bswap32(num_elements);

        std::vector<uint8_t> labels(num_elements);
        file.read(reinterpret_cast<char*>(labels.data()), labels.size());
        file.close();
        return labels;
    }

    // Function to convert images vector to tensor
    torch::Tensor convert_to_tensor(const std::vector<uint8_t>& elements, int32_t num_elements, int32_t num_rows, int32_t num_cols) {
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        torch::Tensor tensor;
        tensor = torch::from_blob(const_cast<uint8_t*>(elements.data()), {num_elements, 1, num_rows, num_cols}, options);
        return tensor.clone(); // Clone to ensure the tensor owns its memory
    }
    // Function to convert labels vector to tensor
    torch::Tensor convert_to_tensor(const std::vector<uint8_t>& elements, int32_t num_elements) {
        auto options = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU);
        torch::Tensor tensor;
        tensor = torch::from_blob(const_cast<uint8_t*>(elements.data()), {num_elements}, options);
        return tensor.clone(); // Clone to ensure the tensor owns its memory
    }
};

#endif