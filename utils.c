#include "utils.h"
#include <math.h>
#include <stdlib.h>


float sigmoid(float x)
{
    return (1 / (1 + expf(-x)));
}

float mse(float *a, float *b, int size)
{
	float error = 0;
	for (int i = 0; i < size; i++) {
		error += pow((b[i] - a[i]), 2);
	}
	return error / size;
}

float relu(float x)
{
    return x < 0 ? 0 : x;
}

float frand(float a)
{
    float x = (float)rand()/(float)(RAND_MAX/a);
    return x;
}