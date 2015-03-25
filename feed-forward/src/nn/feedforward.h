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
  /*
  const T e = exp(-x);
  return (std::isnan(e) ?
          nexttoward(1.0, 0.0) :
          (std::isinf(e) ? nexttoward(0.0, 1.0) : (1.0 / (1.0 + e))));
  */
  return 1.0 / (1.0 + exp(-x));
}

template <typename T>
inline T derivate(const T s) {
  return s * (1 - s);
}

template <typename T, size_t Min, size_t Mout>
struct Layer {
 public:
  using Connections = array<array<T, Min+1>, Mout>;
  using In = array<T, Min>;
  using Out = array<T, Mout>;

  Connections weights;
  Connections gradient;

  Out activations;
  Out outputs;
  Out error;

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

  void updateStates(const In& x) {
    for (size_t j = 0; j < Mout; j++) {
      activations[j] = weights[j][Min];
      for (size_t i = 0; i < Min; i++) {
        activations[j] += weights[j][i] * x[i];
      }
      outputs[j] = sigmoid(activations[j]);
    }
  }

  template <size_t Mnext>
  const Out& backpropagate(const In& x, const array<T, Mnext>& nextError,
                           const array<array<T, Mout+1>, Mnext>& nextWeights) {
    for (size_t j = 0; j < Mout; j++) {
      error[j] = 0;
      for (size_t k = 0; k < Mnext; k++) {
        error[j] += nextWeights[k][j] * nextError[k];
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
    os << "--------------------" << endl;
  }
};

template<typename T, size_t... Ms>
struct FeedForwardLayer;

template<typename T, size_t Min, size_t Mout>
struct FeedForwardLayer<T, Min, Mout> : Layer<T, Min, Mout> {
public:
  using Neurons = Layer<T, Min, Mout>;
  using Connections = array<array<T, Min+1>, Mout>;
  using NetOutput = typename Neurons::Out;
  using In = typename Neurons::In;
  using Out = typename Neurons::Out;

  static const size_t M = Mout;
  static const size_t K = Mout;

  FeedForwardLayer() { }

  // forward computation
  const NetOutput& forward(const In& x) {
    Neurons::updateStates(x);
    return Neurons::outputs;
  }

  // error backpropagation
  const Out& backpropagate(const In& x, const NetOutput& t) {
    for (size_t j = 0; j < K; j++) {
      Neurons::error[j] = (Neurons::outputs[j] - t[j])
        * derivate(Neurons::activations[j]);
      for (size_t i= 0; i < Min; i++) {
        Neurons::gradient[j][i] = Neurons::error[j] * x[i];
      }
      Neurons::gradient[j][Min] = Neurons::error[j];
    }
    return Neurons::error;
  }

  void print(ostream& os) const {
    Neurons::print(os);
  }
};

template<typename T, size_t Min, size_t Mout, size_t... Ms>
struct FeedForwardLayer<T, Min, Mout, Ms...> : Layer<T, Min, Mout>  {
public:
  using Neurons = Layer<T, Min, Mout>;
  using NextLayer = FeedForwardLayer<T, Mout, Ms...>;
  using In = typename Neurons::In;
  using Out = typename Neurons::Out;
  using NetOutput = typename NextLayer::NetOutput;

  static const size_t K = NextLayer::K;
  static const size_t D = Min;
  static const size_t M = Mout;

  NextLayer nextLayer;

  const NetOutput& forward(const In& x) {
    Neurons::updateStates(x);
    return nextLayer.forward(Neurons::outputs);
  }

  const Out& backpropagate(const In& x, const NetOutput& t) {
    Neurons::backpropagate(x,
                           nextLayer.backpropagate(Neurons::outputs, t),
                           nextLayer.weights);
    return Neurons::error;
  }

  void print(ostream& os) const {
    Neurons::print(os);
    nextLayer.print(os);
  }
};

#endif
