// Copyright (C) 2015 Tudor Berariu

#ifndef __READ_MNIST_H__
#define __READ_MNIST_H__

#include <cstring>

#include <array>
#include <algorithm>

#include "../general/dataset.h"
#include "../idx-parser/idx_parser.h"

using namespace std;

template<typename T>
using MNIST = Dataset<T, 54000, 6000, 10000, 784, 10>;

template<typename T>
void readMnistDataSet(const char* folder, MNIST<T>*& mnist) {
  using _MNIST = MNIST<T>;
  mnist = new _MNIST();

  array<size_t, _MNIST::NTrain + _MNIST::NValid> indices1;
  for (size_t n = 0; n < _MNIST::NTrain + _MNIST::NValid; n++)
    indices1[n] = n;
  random_shuffle(indices1.begin(), indices1.end());

  array<size_t, _MNIST::NTest> indices2;
  for (size_t n = 0; n < _MNIST::NTest; n++)
    indices2[n] = n;

  array<array<T, _MNIST::D>, 0> xDummy;
  array<array<T, _MNIST::K>, 0> tDummy;

  readIdx3("dataset/train-images-idx3-ubyte",
           mnist->xTrain, mnist->xValid, xDummy, indices1);
  readIdx1("dataset/train-labels-idx1-ubyte",
           mnist->tTrain, mnist->tValid, tDummy, indices1);

  readIdx3("dataset/t10k-images-idx3-ubyte",
           xDummy, xDummy, mnist->xTest, indices2);
  readIdx1("dataset/t10k-labels-idx1-ubyte",
           tDummy, tDummy, mnist->tTest, indices2);
}

#endif
