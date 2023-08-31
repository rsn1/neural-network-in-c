#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include "matrix.h"
#include "utils.h"

#define N_SAMPLES 20
#define N_LAYERS 3 //number of layers
#define N_WEIGHTS N_LAYERS-1

//n_nodes: array of ints describing number of nodes per layer
int n_nodes[N_LAYERS] = {1,4,1};

//wih 1 input : Wx+b ->  W : 4x1 ; x: 1x1  -> Wx = 4x1
//wih 2 inputs : Wx +b -> W: 4x2 ; x : 2x1 -> Wx = 4x1
//who : Wx+b -> W 1x4 x: 4x1

//dummy test data
float train_x[N_SAMPLES] = {1,2,3,4,5,6,7,8,9,10};
float train_y[N_SAMPLES] = {1,4,9,16,25,36,49,64,81,100};

typedef struct {
    Matrix *biases;
    Matrix *weights;
    Matrix *outputs;
    int size;
} Network;

Network network_alloc()
{
    Network net;
    net.biases = malloc(sizeof(Matrix)*N_WEIGHTS);
    net.weights = malloc(sizeof(Matrix)*N_WEIGHTS);
    net.outputs = malloc(sizeof(Matrix)*N_WEIGHTS);
    assert(net.biases != NULL);
    assert(net.weights != NULL);
    assert(net.outputs != NULL);
    net.size = N_WEIGHTS;
    //biases[0] = (4,1)
    //weights[0] = (4,1)
    //outputs[0] = (4,1)

    //biases[1] = 1,1
    //weights[1] = 1,4
    //outputs[1] = 1,1
    //1. Wx + b = (4,1)(1,1)+(4,1) = (4,1)
    //2. Wx + b = (1,4)(4,1)+(1,1) = (1,1)
    for (int i = 0; i < N_WEIGHTS; ++i) {
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
    for (int i = 1; i < N_WEIGHTS; ++i) {
        Matrix prev_output = net->outputs[i-1];
        matrix_mul(&net->outputs[i],&net->weights[i],&prev_output); //W_i * x_(i-1) 
        matrix_add(&net->outputs[i],&net->outputs[i],&net->biases[i]);
        matrix_act_func(&net->outputs[i],&net->outputs[i],relu);
    }
}

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

//W: 1x4 1,1
int main(void) 
{
    srand(time(0));

    //matrix tests
    Matrix d = matrix_alloc(2,2);
    matrix_rand(&d,-10,10);
    printf("d matrix: \n");
    matrix_print(&d);
    printf("------ \n");
    printf("d matrix after add: \n");
    matrix_add(&d,&d,&d);
    matrix_print(&d);
    printf("------ \n");
    printf("d matrix after relu \n");
    matrix_act_func(&d,&d,relu);
    matrix_print(&d);
    printf("------ \n");
  
    //network tests
    Network net = network_alloc();

    Matrix input = matrix_alloc(1,1);
    input.data[0] = 1.0;

    network_forward(&net,&input);
    matrix_print(&net.outputs[0]);
    printf("-------- \n");
    matrix_print(&net.outputs[1]);
    //N_WEIGHTS-1
    network_free(&net);
    return 0;
}


