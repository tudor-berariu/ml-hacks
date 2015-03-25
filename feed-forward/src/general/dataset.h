// Copyright (C) 2015 Tudor Berariu

#ifndef __DATASET_H__
#define __DATASET_H__

#include <array>

using namespace std;

template<typename T, size_t Ntrain, size_t Nvalid, size_t Ntest,
         size_t Din, size_t Dout>
struct Dataset {
  using In = array<T, Din>;
  using Out = array<T, Dout>;

  static const size_t N = Ntrain + Nvalid + Ntest;
  static const size_t NTrain = Ntrain;
  static const size_t NValid = Nvalid;
  static const size_t NTest = Ntest;
  static const size_t D = Din;
  static const size_t K = Dout;

  array<In, Ntrain> xTrain;
  array<In, Nvalid> xValid;
  array<In, Ntest> xTest;
  array<Out, Ntrain> tTrain;
  array<Out, Nvalid> tValid;
  array<Out, NTest> tTest;
};

#endif
