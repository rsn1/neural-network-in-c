#include <stdio.h>
//#include <math.h>
#include "utils.h"
#include <stdlib.h>


#define N_SAMPLES 20
#define N_LAYERS 3

int n_nodes[N_LAYERS] = {4,6,4};

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

Layer *network;

int main(void) 
{
    printf("Hello\n");


    float x = 5.0;
    float res = sigmoid(x);
    printf("%f \n",res);

    printf("%f \n",relu(x));
    x = -2.0;
    printf("%f",relu(x));

    create_network(network);


    return 0;
}


//n_nodes: array of ints describing number of nodes per layer
void create_network(Layer *network)
{
    network = malloc(N_LAYERS*sizeof(Layer));
    for (int i = 0; i < N_LAYERS; ++i) {
        network[i].n_nodes = n_nodes[i];
        network[i].nodes = malloc(n_nodes[i] * sizeof(int));
        for (int j = 0; j < n_nodes[i]; ++j) {
            network[i].nodes[j].value = 0;
            network[i].nodes[j].bias = 0;
        }
    }
}

void free_network(Layer *network)
{
    for (int i = 0; i < N_LAYERS; ++i) {
        free(network[i].nodes);
    }
    free(network);
}


void forward(void)
{
    for (int i = 0; i < N_LAYERS; ++i) {
        for (int j = 0; j < n_nodes[i]; ++j) {





        }
    }
}