#include "utils.h"
#include <math.h>

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