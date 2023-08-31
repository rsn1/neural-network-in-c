#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "network.h"


Network network_alloc(int n_nodes[], int n_weights)
{
    Network net;
    net.biases = malloc(sizeof(Matrix)*n_weights);
    net.weights = malloc(sizeof(Matrix)*n_weights);
    net.outputs = malloc(sizeof(Matrix)*n_weights);
    assert(net.biases != NULL);
    assert(net.weights != NULL);
    assert(net.outputs != NULL);
    net.size = n_weights;
    //biases[0] = (4,1)
    //weights[0] = (4,1)
    //outputs[0] = (4,1)

    //biases[1] = 1,1
    //weights[1] = 1,4
    //outputs[1] = 1,1
    //1. Wx + b = (4,1)(1,1)+(4,1) = (4,1)
    //2. Wx + b = (1,4)(4,1)+(1,1) = (1,1)
    for (int i = 0; i < n_weights; ++i) {
        net.biases[i] = matrix_alloc(n_nodes[i+1],1); 
        net.weights[i] = matrix_alloc(n_nodes[i+1],n_nodes[i]);
        net.outputs[i] = matrix_alloc(n_nodes[i+1],1);
        //randomize initial weights
        matrix_rand(&net.biases[i],-1,1);
        matrix_rand(&net.weights[i],-1,1);
    }
    return net;
}

void network_forward(Network *net, Matrix *input)
{
    assert(input->rows==net->weights[0].cols);
    matrix_mul(&net->outputs[0],&net->weights[0],input); //o = Wx
    matrix_add(&net->outputs[0],&net->outputs[0],&net->biases[0]); //o = o + b
    matrix_act_func(&net->outputs[0],&net->outputs[0],relu); //o = act(o)
    //loop through hidden layers
    for (int i = 1; i < net->size; ++i) {
        Matrix prev_output = net->outputs[i-1];
        matrix_mul(&net->outputs[i],&net->weights[i],&prev_output); //W_i * x_(i-1) 
        matrix_add(&net->outputs[i],&net->outputs[i],&net->biases[i]);
        matrix_act_func(&net->outputs[i],&net->outputs[i],relu);
    }
}

void network_backward(Network *net) {}

void network_free(Network *net)
{
    if (net != NULL) {
        for (int i = 0; i < net->size; ++i) {
            free(net->weights[i].data);
            free(net->biases[i].data);
            free(net->outputs[i].data);
        }
        free(net->biases);
        free(net->outputs);
        free(net->weights);
    }
}