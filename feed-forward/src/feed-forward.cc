// Copyright (C) 2015 Tudor Berariu

#include "nn/feedforward.h"
#include "read-mnist/read_mnist.h"

#include <cmath>
#include <fstream>
#include <array>
#include <fenv.h>

#include <typeinfo>

using namespace std;

int main(int argc, char* argv[])
{
  feenableexcept(FE_INVALID | FE_OVERFLOW);
  using Mnist = MNIST<long double>;
  using NN = Layer<long double, Mnist::D, 300, Mnist::K>;
  Mnist* mnist;
  readMnistDataSet(argv[1], mnist);

  cout << "done" << endl;
  NN nn;
  for (int i = 0; i < 20; i++) {
    auto out = nn.forward(mnist->xTrain[1]);
    long double error = 0.0;
    for (auto j = 0; j < 10; j++) {
      error += (out[j] - mnist->tTrain[1][j]) * (out[j] - mnist->tTrain[1][j]);
    }
    cout << error << endl;
    nn.backpropagate(mnist->xTrain[0], mnist->tTrain[0]);
    nn.adjust(0.5);
  }

  delete(mnist);
  return 0;
}
