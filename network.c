#include <stdlib.h>
#include <assert.h>
#include "utils.h"
#include "network.h"
#include "matrix.h"
#include <stdio.h>
#define LEARNING_RATE 0.3

Network network_alloc(int n_nodes[], int n_weights)
{
    Network net;
    net.biases = malloc(sizeof(Matrix)*n_weights);
    net.weights = malloc(sizeof(Matrix)*n_weights);
    net.outputs = malloc(sizeof(Matrix)*n_weights);
    net.d_biases = malloc(sizeof(Matrix)*n_weights);
    net.d_weights = malloc(sizeof(Matrix)*n_weights);
    assert(net.biases != NULL);
    assert(net.weights != NULL);
    assert(net.outputs != NULL);
    assert(net.d_biases != NULL);
    assert(net.d_weights != NULL);
    //size = number of weight matrices
    net.size = n_weights;
    //biases[0] = (4,1)
    //weights[0] = (4,1)
    //outputs[0] = (4,1)

    //biases[1] = 1,1
    //weights[1] = 1,4
    //outputs[1] = 1,1
    //1. Wx + b = (4,1)(1,1)+(4,1) = (4,1)
    //2. Wx + b = (1,4)(4,1)+(1,1) = (1,1)
    //[4,5,3,2]
    for (int i = 0; i < n_weights; ++i) {
        net.biases[i] = matrix_alloc(n_nodes[i+1],1);
        net.d_biases[i] = matrix_alloc(n_nodes[i+1],1); 
        net.weights[i] = matrix_alloc(n_nodes[i+1],n_nodes[i]);
        net.d_weights[i] = matrix_alloc(n_nodes[i+1],n_nodes[i]);
        net.outputs[i] = matrix_alloc(n_nodes[i+1],1);
        //randomize initial weights
        matrix_rand(&net.biases[i],-1,1);
        matrix_rand(&net.weights[i],-1,1);
    }
    return net;
}

void network_free(Network *net)
{
    if (net != NULL) {
        for (int i = 0; i < net->size; ++i) {
            matrix_free(&(net->weights[i]));
            matrix_free(&(net->biases[i]));
            matrix_free(&(net->outputs[i]));
            matrix_free(&(net->d_biases[i]));
            matrix_free(&(net->d_weights[i]));
        }
        free(net->biases);
        free(net->outputs);
        free(net->weights);
        free(net->d_weights);
        free(net->d_biases);
    }
}

void network_forward(Network *net, Matrix *input)
{
    printf("Forward pass\n");
    assert(input->rows==net->weights[0].cols);
    //input -> first hidden layer
    matrix_mul(&net->outputs[0],&net->weights[0],input); //o = Wx
    matrix_add(&net->outputs[0],&net->outputs[0],&net->biases[0]); //o = o + b
    matrix_elem_func(&net->outputs[0],&net->outputs[0],relu); //o = act(o)
    //loop through hidden layers
    for (int i = 1; i < net->size; ++i) {
        Matrix prev_output = net->outputs[i-1];
        matrix_mul(&net->outputs[i],&net->weights[i],&prev_output); //W_i * x_(i-1) 
        matrix_add(&net->outputs[i],&net->outputs[i],&net->biases[i]);
        matrix_elem_func(&net->outputs[i],&net->outputs[i],relu);
    }
}

//calculate gradients, update weights
void network_backward(Network *net, Matrix* label)
{
    printf("Backward pass\n");
    Matrix net_out = net->outputs[net->size-1];
    assert(net_out.rows == label->rows);
    assert(net_out.cols == label->cols);
    Matrix delta_l = matrix_alloc(net_out.rows,net_out.cols);
    Matrix rhs = matrix_alloc(net_out.rows,net_out.cols);
    matrix_sub(&delta_l,&net_out,label);
    //calculate last output without act func for weight update
    matrix_mul(&rhs,&net->weights[net->size-1],&net->outputs[net->size-2]);
    matrix_elem_func(&rhs,&rhs,relu_d);
    matrix_elem_mul(&delta_l,&delta_l,&rhs);

    Matrix x2t = matrix_alloc(net->outputs[net->size-2].cols,net->outputs[net->size-2].rows);
    matrix_transpose(&x2t,&net->outputs[net->size-2]);
    Matrix dedw = matrix_alloc(net->weights[net->size-1].rows,net->weights[net->size-1].cols);
    matrix_mul(&dedw,&delta_l,&x2t);
    
   

    matrix_free(&delta_l);
    matrix_free(&rhs);
    matrix_free(&x2t);
}



void network_train(Network* net, Matrix *input) {
    network_forward(net,input);
}

