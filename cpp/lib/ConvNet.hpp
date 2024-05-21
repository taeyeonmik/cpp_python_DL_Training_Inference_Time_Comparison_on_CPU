#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>

using namespace std;

class ConvNet : public torch::nn::Module {
private:
    uint8_t k = 3;
    uint8_t p = 1;
    uint8_t s = 1;
    uint8_t poolk = 2;
    uint8_t pools = 1;

    torch::nn::Sequential convBlock1 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 16, /*kernel_size=*/k).stride(s).padding(p)),
            torch::nn::BatchNorm2d(16),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(poolk).stride(pools))
    };
    torch::nn::Sequential convBlock2 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 32, /*kernel_size=*/k).stride(s).padding(p)),
            torch::nn::BatchNorm2d(32),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(poolk).stride(pools))
    };
    torch::nn::Sequential convBlock3 {
            torch::nn::Conv2d(torch::nn::Conv2dOptions(32, 64, /*kernel_size=*/k).stride(s).padding(p)),
            torch::nn::BatchNorm2d(64),
            torch::nn::ReLU(),
            torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(poolk).stride(pools))
    };
    torch::nn::Linear fc;
public:
    // default constructor
    explicit ConvNet(uint16_t numClasses, uint8_t wout, uint8_t hout) : fc(wout*hout*64, numClasses) {
        // register_module() is needed when using parameters() method later on
        register_module("convBlock1", convBlock1);
        register_module("convBlock2", convBlock2);
        register_module("convBlock3", convBlock3);
        register_module("fc", fc);
        cout << "ConvNet is ready to train." << endl;
    }
    // forward
    torch::Tensor forward(torch::Tensor x){
        x = convBlock1->forward(x);
        x = convBlock2->forward(x);
        x = convBlock3->forward(x);
        x = x.view({x.size(0), -1}); // flatten x
        return fc->forward(x);
    }
};

