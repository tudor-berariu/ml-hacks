// Copyright (C) 2015 Tudor Berariu

#include "idx_parser.h"

int reverseBytes(const int i) {
  unsigned char byte1, byte2, byte3, byte4;
  byte1 =  i        & 255;
  byte2 = (i >>  8) & 255;
  byte3 = (i >> 16) & 255;
  byte4 = (i >> 24) & 255;
  return ((int)byte1 << 24) + ((int)byte2 << 16) + ((int)byte3 << 8) + byte4;
}
