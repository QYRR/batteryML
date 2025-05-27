#ifndef __ENSEMBLE_H__
#define __ENSEMBLE_H__
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define L1

// Constants definition
#define LEAF_INDICATOR 10
#define N_ESTIMATORS 443
#define N_TREES 443
#define LEAF_LENGTH 1
#define N_NODES 15447
#define INPUT_LENGTH 10
#define N_LEAVES 0
#define OUTPUT_LENGTH 1

// SIMD
typedef uint8_t v4u __attribute__((vector_size(4)));
typedef uint16_t v2u __attribute__((vector_size(4)));
#define ADD(x, y) ((x) + (y))

#endif //__ENSEMBLE_H__