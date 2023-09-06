#include <math.h>
#include <stdlib.h>
#include "utils.h"
#include <assert.h>
#include "matrix.h"

float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

float relu(float x)
{
    return x < 0 ? 0 : x;
}

float relu_d(float x)
{
    return x > 0 ? 1 : 0;
}

float frand(void)
{
    return (float) rand() / (float) RAND_MAX;
}