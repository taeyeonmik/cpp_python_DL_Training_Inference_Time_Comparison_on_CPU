#include <iostream>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>

#include "ConvNet.hpp"

using namespace std;

class ConvNetClassifier {
private:
public:
    torch::Device *device;
    ConvNet *model;
    torch::optim::Adam *optimizer;
    vector<double> loss;
    vector<double> accuracy;

    // counstructor
    ConvNetClassifier(int64_t numClasses, double learningRate) {
        device = new torch::Device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        cout << "ConvNet Classifier is being compiled...\n numClasses: " << numClasses << "\n" << "learningRate: " << learningRate << "\n" << "optimizer: Adam" << endl;
        model = new ConvNet(numClasses);
        (*model).to(*device);
        optimizer = new torch::optim::Adam(
                (*model).parameters(),
                torch::optim::AdamOptions(learningRate)
        );
    }
    void train() {

    }
    void evaluate() {

    }

};