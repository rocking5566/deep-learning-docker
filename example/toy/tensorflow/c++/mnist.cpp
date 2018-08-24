#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#include <fstream>
using namespace std;
using namespace tensorflow;
using namespace cv;

// Refer to https://gist.github.com/kyrs/9adf86366e9e4f04addb

GraphDef LoadGraph(const string modelPath)
{
    GraphDef network;
    Status status = ReadBinaryProto(Env::Default(), modelPath, &network);
    if (!status.ok())
    {
        std::cout << status.ToString() << endl;
        throw 1;
    }

    return move(network);
}

Session* CreateSession(const string modelPath)
{
    GraphDef network = LoadGraph(modelPath);

    // Initialize a tensorflow session
    Session* pSession = NULL;
    Status status = NewSession(SessionOptions(), &pSession);
    if (!status.ok())
    {
        std::cout << status.ToString() << endl;
        throw 2;
    }

    // Add the graph to the session
    status = pSession->Create(network);
    if (!status.ok()) {
        std::cout << status.ToString() << endl;
        throw 3;
    }

    return pSession;
}

// Free any resources used by the session
void DestroySession(Session* pSession)
{
    pSession->Close();
}

Tensor MatToTensor(const Mat& img, bool bNormalizeInput = true)
{
    // TODO - find a way to convert data faster
    int height = img.rows;
    int width = img.cols;
    Tensor retTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, height * width}));
    auto retTensorMapped = retTensor.tensor<float, 2>();    // 0: index of data/image, 1: data/image content (gray scale)

    Mat floatImg;
    img.convertTo(floatImg, CV_32FC1);

    if (bNormalizeInput)
    {
        floatImg = floatImg / 255;
    }

    const float * source_data = (float*) floatImg.data;

    int index = 0;
    for (int y = 0; y < height; ++y)
    {
        const float* source_row = source_data + y * width;
        for (int x = 0; x < width; ++x)
        {
            retTensorMapped(0, index++) = *(source_row + x);    // only inference one data/image
        }
    }

    return move(retTensor);
}

Tensor PredictMnist(Session* pSession, Tensor& digitTensor)
{
    // running the graph
    vector<pair<string, Tensor>> inputs = {{"dense_1_input:0", digitTensor }};
    vector<Tensor> preds;
    Status status = pSession->Run(inputs, {"pred/Softmax:0"}, {}, &preds);
    if (!status.ok())
    {
        cout << status.ToString() << endl;
        throw 4;
    }

    return move(preds.at(0));
}

void ShowResult(const Mat& digit, const Tensor& pred)
{
    auto scores = pred.flat<float>();
    for (int i = 0; i < 10; ++i)
    {
        cout << i << " - " << scores(i) << endl;
    }

    resize(digit, digit, Size(500, 500));
    imshow("digit", digit);
    waitKey(0);
}

int main(int argc, char* argv[])
{
    // model.pb is trained from keras_to_tensorflow_mnist.ipynb
    // You can also download pretrained model from the following link
    // https://drive.google.com/file/d/18oLp5avejnftsoFkTz6GIuDsxVh8GtTn/view?usp=sharing
    Session* pSession = CreateSession("model.pb");

    Mat digit = cv::imread("4.jpg", 0);
    Tensor digitTensor = MatToTensor(digit);
    Tensor pred = PredictMnist(pSession, digitTensor);
    DestroySession(pSession);

    ShowResult(digit, pred);

    return 0;
}