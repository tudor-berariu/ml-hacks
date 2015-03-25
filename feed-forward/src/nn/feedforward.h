// Copyright (C) 2015 Tudor Berariu

#ifndef __FEEDFORWARD_H__
#define __FEEDFORWARD_H__

#include <cmath>

#include <array>
#include <random>
#include <fstream>
#include <iostream>

#include <fenv.h>

using namespace std;

template <typename T>
inline T sigmoid(const T x) {
  feenableexcept(0);
  const T e = exp(-x);
  feenableexcept(FE_INVALID | FE_OVERFLOW);
  return (std::isnan(e) ?
          nexttoward(1.0, 0.0) :
          (std::isinf(e) ? nexttoward(0.0, 1.0) : (1.0 / (1.0 + e))));
}

template <typename T>
inline T derivate(const T s) {
  return s * (1 - s);
}

template<typename T, size_t... Ms>
struct Layer;

template<typename T, size_t Min, size_t Mout>
struct Layer<T, Min, Mout> {
public:
  using In = array<T, Min>;
  using Out = array<T, Mout>;
  using Connections = array<array<T, Min+1>, Mout>;
  using NetOutput = Out;

  Out activations;
  Out outputs;
  Out error;
  Connections gradient;

  static const size_t M = Mout;
  static const size_t K = Mout;

  Connections weights;

  // initialize the layers
  Layer() {
    random_device rd;
    default_random_engine e(rd());
    uniform_real_distribution<T> next(-0.05, 0.05);
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        weights[j][i] = next(e);
      }
    }
  }

  // forward computation
  const NetOutput& forward(const In& x) {
    for (size_t j = 0; j < Mout; j++) {
      activations[j] = weights[j][Min];
      for (size_t i = 0; i < Min; i++) {
        activations[j] += weights[j][i] * x[i];
      }
      outputs[j] = sigmoid<T>(activations[j]);
    }
    return outputs;
  }

  // error backpropagation
  const Out& backpropagate(const In& x, const NetOutput& t) {
    for (size_t j = 0; j < K; j++) {
      error[j] = (outputs[j] - t[j]) * derivate(activations[j]);
      for (size_t i= 0; i < Min; i++) {
        gradient[j][i] = error[j] * x[i];
      }
      gradient[j][Min] = error[j];
    }
    return error;
  }

  void adjust(const T learningRate) {
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        weights[j][i] -= gradient[j][i] * learningRate;
      }
    }
  }

  void print(ostream& os) const {
    os << "--------------------" << endl;
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        os << weights[j][i] << " ";
      }
      os << endl;
    }
    os << "--------------------" << endl;
  }
};

template<typename T, size_t Min, size_t Mout, size_t... Ms>
struct Layer<T, Min, Mout, Ms...> {
public:
  using NextLayer = Layer<T, Mout, Ms...>;
  using In = array<T, Min>;
  using Out = array<T, Mout>;
  using Connections = array<array<T, Min+1>, Mout>;
  using NetOutput = typename NextLayer::NetOutput;

  static const size_t K = NextLayer::K;
  static const size_t M = Mout;

  NextLayer nextLayer;

  Out activations;
  Out outputs;

  Out error;
  Connections gradient;

  Connections weights;

  Layer() : nextLayer { } {
    random_device rd;
    default_random_engine e(rd());
    uniform_real_distribution<T> next(-0.05, 0.05);
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        weights[j][i] = next(e);
      }
    }
  }

  const NetOutput& forward(const In& x) {
    for (size_t j = 0; j < Mout; j++) {
      activations[j] = weights[j][Min];
      for (size_t i = 0; i < Min; i++) {
        activations[j] += weights[j][i] * x[i];
      }
      outputs[j] = sigmoid<T>(activations[j]);
    }
    return nextLayer.forward(outputs);
  }

  const Out& backpropagate(const In& x, const NetOutput& t) {
    const array<T, NextLayer::M>& nextError =
      nextLayer.backpropagate(outputs, t);
    for (size_t j = 0; j < Mout; j++) {
      error[j] = 0;
      for (size_t k = 0; k < nextLayer.M; k++) {
        error[j] += nextLayer.weights[k][j] * nextError[k];
      }
      error[j] *= derivate(activations[j]);
      for (size_t i = 0; i < Min; i++) {
        gradient[j][i] = x[i] * error[j];
      }
      gradient[j][Min] = error[j];
    }
    return error;
  }

  void adjust(const T learningRate) {
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        weights[j][i] -= gradient[j][i] * learningRate;
      }
    }
  }

  void print(ostream& os) const {
    os << "--------------------" << endl;
    for (size_t j = 0; j < Mout; j++) {
      for (size_t i = 0; i <= Min; i++) {
        os << weights[j][i] << " ";
      }
      os << endl;
    }
    os << " --- " << endl;
    nextLayer.print(os);
  }
};

#endif
