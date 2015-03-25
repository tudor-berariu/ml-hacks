// Copyright (C) 2015 Tudor Berariu

#ifndef __IDX_PARSER_H__
#define __IDX_PARSER_H__

#include <cassert>

#include <array>
#include <iostream>
#include <fstream>

using namespace std;

int reverseBytes(const int i);

template<typename T, size_t Ntrain, size_t Nvalid, size_t Ntest, size_t D>
void readIdx3(const char* fileName,
              array<array<T, D>, Ntrain>& trainData,
              array<array<T, D>, Nvalid>& validData,
              array<array<T, D>, Ntest>& testData,
              const array<size_t, Ntrain+Nvalid+Ntest>& indices) {
  int magicNumber, matricesNo, rowsNo, colsNo;
  ifstream infile(fileName, ios::binary);
  assert(infile.is_open());
  infile.read((char*)&magicNumber, sizeof(magicNumber));
  magicNumber = reverseBytes(magicNumber);

  infile.read((char*)&matricesNo, sizeof(matricesNo));
  matricesNo= reverseBytes(matricesNo);
  assert(matricesNo == (Ntrain + Nvalid + Ntest));

  infile.read((char*)&rowsNo, sizeof(rowsNo));
  rowsNo= reverseBytes(rowsNo);
  infile.read((char*)&colsNo, sizeof(colsNo));
  colsNo= reverseBytes(colsNo);
  assert(rowsNo * colsNo == D);

  for (int n = 0; n < matricesNo; n++) {
    if (indices[n] < Ntrain) {
      for (size_t j = 0; j < D; j++) {
        unsigned char temp = 0;
        infile.read((char*)&temp, sizeof(temp));
        trainData[indices[n]][j] = (T)temp;
      }
    } else if (indices[n] < Ntrain + Nvalid) {
      for (size_t j = 0; j < D; j++) {
        unsigned char temp = 0;
        infile.read((char*)&temp, sizeof(temp));
        validData[indices[n]-Ntrain][j] = (T)temp;
      }
    } else {
      for (size_t j = 0; j < D; j++) {
        unsigned char temp = 0;
        infile.read((char*)&temp, sizeof(temp));
        testData[indices[n]-Nvalid-Ntrain][j] = (T)temp;
      }
    }
  }
  infile.close();
}

template<typename T, size_t Ntrain, size_t Nvalid, size_t Ntest, size_t K>
void readIdx1(const char* fileName,
              array<array<T, K>, Ntrain>& trainData,
              array<array<T, K>, Nvalid>& validData,
              array<array<T, K>, Ntest>& testData,
              const array<size_t, Ntrain+Nvalid+Ntest>& indices) {
  int magicNumber, matricesNo;
  ifstream infile(fileName, ios::binary);
  assert(infile.is_open());

  infile.read((char*)&magicNumber, sizeof(magicNumber));
  magicNumber = reverseBytes(magicNumber);

  infile.read((char*)&matricesNo, sizeof(matricesNo));
  matricesNo= reverseBytes(matricesNo);
  assert(matricesNo == (Ntrain + Nvalid + Ntest));

  for(int n = 0; n < matricesNo; n++) {
    unsigned char temp = 0;
    infile.read((char*)&temp, sizeof(temp));

    if (indices[n] < Ntrain) {
      for (size_t k = 0; k < K; k++) {
        trainData[indices[n]][k] = 0.0;
      }
      trainData[indices[n]][temp] = 1.0;
    } else if(indices[n] < Ntrain + Nvalid) {
      for (size_t k = 0; k < K; k++) {
        validData[indices[n]-Ntrain][k] = 0.0;
      }
      validData[indices[n]-Ntrain][temp] = 1.0;
    } else {
      for (size_t k = 0; k < K; k++) {
        testData[indices[n]-Ntrain-Nvalid][k] = 0.0;
      }
      testData[indices[n]-Ntrain-Nvalid][temp] = 1.0;
    }
  }
  infile.close();
}

#endif
