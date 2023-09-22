#ifndef NETWORK_H
#define NETWORK_H
#include "matrix.h"

typedef struct Network {
    Matrix *biases;
    Matrix *weights;
    Matrix *d_biases;
    Matrix *d_weights;
    Matrix *outputs;
    int size;
} Network;

Network network_alloc(int n_nodes[], int n_weights);
void network_forward(Network *net, Matrix *input);
void network_free(Network *net);
void network_backward(Network *net, Matrix *label);
void network_step(Network *net);
float network_loss(Network *net, Matrix *label);

#endif