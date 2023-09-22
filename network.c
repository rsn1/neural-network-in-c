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
    net.outputs = malloc(sizeof(Matrix)*(n_weights+1));//input layer aswell here
    net.d_biases = malloc(sizeof(Matrix)*n_weights);
    net.d_weights = malloc(sizeof(Matrix)*n_weights);
    assert(net.biases != NULL);
    assert(net.weights != NULL);
    assert(net.outputs != NULL);
    assert(net.d_biases != NULL);
    assert(net.d_weights != NULL);
    //size = number of weight matrices
    net.size = n_weights;
    net.outputs[0] = matrix_alloc(n_nodes[0],1);
    for (int i = 0; i < n_weights; ++i) {
        net.biases[i] = matrix_alloc(n_nodes[i+1],1);
        net.d_biases[i] = matrix_alloc(n_nodes[i+1],1); 
        net.weights[i] = matrix_alloc(n_nodes[i+1],n_nodes[i]);
        net.d_weights[i] = matrix_alloc(n_nodes[i+1],n_nodes[i]);
        net.outputs[i+1] = matrix_alloc(n_nodes[i+1],1);
        //randomize initial weights
        matrix_rand(&net.biases[i],-1,1);
        matrix_rand(&net.weights[i],-1,1);
    }
    return net;
}

void network_free(Network *net)
{
    if (net != NULL) {
        matrix_free(&(net->outputs[0]));
        for (int i = 0; i < net->size; ++i) {
            matrix_free(&(net->weights[i]));
            matrix_free(&(net->biases[i]));
            matrix_free(&(net->outputs[i+1]));
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

float network_loss(Network *net, Matrix *label)
{
    Matrix output = net->outputs[net->size];
    assert(output.cols == 1);
    assert(label->cols == 1);
    assert(output.rows == label->rows);
    Matrix out = matrix_alloc(output.rows,output.cols);
    matrix_sub(&out,&output,label);
    return matrix_squared_l2_norm(&out);
}


void network_forward(Network *net, Matrix *input)
{
    printf("Forward pass\n");
    assert(input->rows==net->weights[0].cols);
    //set output of input layer
    matrix_copy(&net->outputs[0],input);
    //input -> first hidden layer
    matrix_mul(&net->outputs[1],&net->weights[0],input); //o = Wx
    matrix_add(&net->outputs[1],&net->outputs[1],&net->biases[0]); //o = o + b
    matrix_elem_func(&net->outputs[1],&net->outputs[1],relu); //o = act(o)
    //loop through hidden layers
    for (int i = 1; i < net->size; ++i) {
        Matrix prev_output = net->outputs[i];
        matrix_mul(&net->outputs[i+1],&net->weights[i],&prev_output); //W_i * x_(i-1) 
        matrix_add(&net->outputs[i+1],&net->outputs[i+1],&net->biases[i]);
        matrix_elem_func(&net->outputs[i+1],&net->outputs[i+1],relu);
    }
}

//calculate gradients, update weights
//L = 2
void network_backward(Network *net, Matrix* label)
{
    printf("Backward pass\n");
    Matrix net_out = net->outputs[net->size];
    assert(net_out.rows == label->rows);
    assert(net_out.cols == label->cols);

    Matrix delta_l = matrix_alloc(net_out.rows,net_out.cols);
    Matrix rhs = matrix_alloc(net_out.rows,1);
    
    matrix_sub(&delta_l,&net_out,label);
    //calculate last output without act func for weight update
    matrix_mul(&rhs,&net->weights[net->size-1],&net->outputs[net->size-1]); //W_3 x X_2
    matrix_elem_func(&rhs,&rhs,relu_d);
    matrix_elem_mul(&delta_l,&delta_l,&rhs);

    Matrix x2t = matrix_alloc(net->outputs[net->size-1].cols,net->outputs[net->size-1].rows);
    matrix_transpose(&x2t,&net->outputs[net->size-1]);
    Matrix dedw3 = matrix_alloc(net->weights[net->size-1].rows,net->weights[net->size-1].cols);
    matrix_mul(&dedw3,&delta_l,&x2t);
    net->d_weights[net->size-1] = dedw3;


    Matrix deltas[net->size]; //0,1,2
    deltas[net->size-1] = delta_l;
    for(int i = net->size-2; i >= 0; --i) { //net size = 3, i=1
        Matrix delta_i = matrix_alloc(net->weights[i+1].cols,1); //cols
        Matrix rhs_i = matrix_alloc(net->weights[i].rows,1);
        Matrix w_t = matrix_alloc(net->weights[i+1].cols,net->weights[i+1].rows);
        matrix_transpose(&w_t,&net->weights[i+1]); //w^T
        matrix_mul(&delta_i,&w_t,&deltas[i+1]); //w_(i+1)^T @ delta_(i+1) assert fail
        matrix_mul(&rhs_i,&net->weights[i],&net->outputs[i]); //W_i x_(i-1)
        matrix_elem_func(&rhs_i,&rhs_i,relu_d); //relu
        matrix_elem_mul(&delta_i,&delta_i,&rhs_i); //W_(i+1)^T delta_(i+1) * relu(W_i x_(i-1))
        
        Matrix x_t = matrix_alloc(net->outputs[i].cols,net->outputs[i].rows);
        matrix_transpose(&x_t,&net->outputs[i]);
        matrix_mul(&net->d_weights[i],&delta_i,&x_t); //store derivatives
        //todo store bias derivatives
        

        deltas[i] = delta_i;
        matrix_free(&rhs_i);
        matrix_free(&w_t);
        matrix_free(&x_t);
    }

    matrix_free(&delta_l);
    matrix_free(&rhs);
    matrix_free(&x2t);
}

void network_step(Network *net)
{
    //update weights
    printf("Updating weights\n");
    for (int i = 0; i < net->size; ++i) {
        Matrix weights = net->weights[i];
        Matrix learn_matrix = matrix_alloc(weights.rows,weights.cols);
        matrix_scalar_mul(&learn_matrix,LEARNING_RATE,&net->d_weights[i]); //multiply derivatives by learning rate
        matrix_sub(&net->weights[i],&weights,&learn_matrix);
        matrix_free(&learn_matrix);
    }
}



