#include <stdio.h>
//#include <math.h>
#include "utils.h"
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include "matrix.h"


#define N_SAMPLES 20
#define N_LAYERS 3 //number of layers
#define N_WEIGHTS N_LAYERS-1

int n_nodes[N_LAYERS] = {1,4,1};
//{n_in,n_hidden1,n_hidden2 ... n_out}


//wih 1 input : Wx+b ->  W : 4x1 ; x: 1x1  -> Wx = 4x1
//wih 2 inputs : Wx +b -> W: 4x2 ; x : 2x1 -> Wx = 4x1
//who : Wx+b ->  x: 1x4
//W : n_hidden x n_input
//x : n_input x 1
//B,F ... N,F
//create_net(n_hidden,)

//input N   NxN_h1 N_h1xN_h2
float train_x[N_SAMPLES] = {1,2,3,4,5,6,7,8,9,10};
float train_y[N_SAMPLES] = {1,4,9,16,25,36,49,64,81,100};


typedef struct {
    float *w_in;
    int n_in;
    float bias;
} Node;


typedef struct {
    Matrix *biases;
    Matrix *weights;
    Matrix *outputs;
} Network;


Network network_alloc()
{
    Network net;
    net.biases = malloc(sizeof(Matrix)*N_WEIGHTS);
    net.weights = malloc(sizeof(Matrix)*N_WEIGHTS);
    net.outputs = malloc(sizeof(Matrix)*N_WEIGHTS);
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
    //loop through hidden layers
    for (int i = 1; i < N_WEIGHTS; ++i) {

    }
}

//W: 1x4 1,1
int main(void) 
{
    srand(time(0));

    Matrix out = matrix_alloc(2,2);
    Matrix d = matrix_alloc(2,2);
    Matrix k = matrix_alloc(3,3);
    //printf("%d \n\n",sizeof d);
   // printf("%d \n",sizeof(Matrix*));
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
    //matrix_mul(&out,&d,&k);
    matrix_print(&out);
    float *p;
    Matrix m;
    m.cols = 8;
    m.rows = 8;
    m.data = malloc(sizeof(float)*8*8);
    printf("%d \n",sizeof(m));
    printf("%d \n",sizeof(&m));
    printf("%d %d %d",sizeof(int), sizeof(float), sizeof(p));
 //   create_network(network);

    return 0;
}


//n_nodes: array of ints describing number of nodes per layer


