#include <stdio.h>
//#include <math.h>
#include "utils.h"
#include <stdlib.h>


#define N_SAMPLES 20


int n_nodes = [4,6,4];

//B,F ... N,F
//create_net(n_hidden,)

//input N   NxN_h1 N_h1xN_h2
float train_x[N_SAMPLES] = {1,2,3,4,5,6,7,8,9,10};
float train_y[N_SAMPLES] = {1,4,9,16,25,36,49,64,81,100};

typedef struct {
    float value;
    float bias;
} Node;


typedef struct {
    int n_nodes;
    Node *nodes;
} Layer;



int main(void) 
{
    printf("Hello\n");


    float x = 5.0;
    float res = sigmoid(x);
    printf("%f \n",res);

    Layer p;


    return 0;
}


//n_nodes: array of ints describing number of nodes per layer
void malloc_network(int n_nodes[], int n_layers)
{
    
}


