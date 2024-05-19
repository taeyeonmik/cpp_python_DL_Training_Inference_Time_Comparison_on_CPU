#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

using namespace std;

class ConvNet : public torch::nn::Module {
private:
    int64_t wh[2] = {0, 0};
    int64_t *whPtr = wh;

    torch::nn::Sequential convBlock1 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, /*kernel_size=*/3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(1))
    };
    torch::nn::Sequential convBlock2 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, /*kernel_size=*/3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(1))
    };
    torch::nn::Sequential convBlock3 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/3).stride(1).padding(1)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(1))
    };
    torch::nn::Linear fc;


protected:
    void getOutputSize(torch::Tensor x) {
        auto size = x.sizes();
        *(whPtr + 0) = size[0];
        *(whPtr + 1) = size[1];
    }
public:
    // default constructor
    explicit ConvNet(int64_t numClasses) : fc(64, numClasses) {
        // register_module() is needed when using parameters() method later on
        register_module("convBlock1", convBlock1);
        register_module("convBlock2", convBlock2);
        register_module("convBlock3", convBlock3);
        register_module("fc", fc);
        cout << "Net created" << endl;
        // calculate final output shape
    }

    // forward
    torch::Tensor forward(torch::Tensor x){
        x = convBlock1->forward(x);
        x = convBlock2->forward(x);
        x = convBlock3->forward(x);
        if (wh[0] == 0) { getOutputSize(x); }
        x = x.view({-1, wh[0] * wh[1] * 64}); // flatten x
        return fc->forward(x);
    }

    // save parameters
    void save();
};

