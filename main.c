#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include "matrix.h"
#include "utils.h"
#include "network.h"

#define N_SAMPLES 20
#define N_LAYERS 4

//B,F
//wih 1 input : Wx+b ->  W : 4x1 ; x: 1x1  -> Wx = 4x1
//wih 2 inputs : Wx +b -> W: 4x2 ; x : 2x1 -> Wx = 4x1
//who : Wx+b -> W 1x4 x: 4x1

//dummy test data
float train_x[N_SAMPLES] = {1,2,3,4,5,6,7,8,9,10};
float train_y[N_SAMPLES] = {1,4,9,16,25,36,49,64,81,100};

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
    matrix_elem_func(&d,&d,relu);
    matrix_print(&d);
    printf("------ \n");
  
    //network tests
    //n_nodes: array of ints describing number of nodes per layer
    int n_nodes[N_LAYERS] = {4,5,3,2};
    Network net = network_alloc(n_nodes,N_LAYERS-1);

    Matrix input = matrix_alloc(4,1);
    matrix_set_elem(&input,0,0,1);
    matrix_set_elem(&input,1,0,2);
    matrix_set_elem(&input,2,0,3);
    matrix_set_elem(&input,3,0,4);
    printf("Input: \n");
    matrix_print(&input);
    Matrix label = matrix_alloc(4,1);
    matrix_set_elem(&label,0,0,1);
    matrix_set_elem(&label,1,0,4);
    matrix_set_elem(&label,2,0,9);
    matrix_set_elem(&label,3,0,16);
    printf("Output: \n");
    matrix_print(&label);
    
    network_forward(&net,&input);
    printf("First weight matrix: \n");
    matrix_print(&net.weights[0]);
    printf("First bias matrix: \n");
    matrix_print(&net.biases[0]);
    printf("First output matrix: \n");
    matrix_print(&net.outputs[1]);
    printf("-------- \n");
    printf("Second output matrix: \n");
    matrix_print(&net.outputs[1]);
    printf("-------- \n");
    printf("Last output matrix: \n");
    matrix_print(&net.outputs[net.size]);
    printf("-------- \n");
    // //N_WEIGHTS-1
    Matrix labels = matrix_alloc(2,1);
    matrix_set_elem(&labels,0,0,1);
    matrix_set_elem(&labels,1,0,2);

    //calculate derivatives
    network_backward(&net,&labels);

    //update weights
    network_step(&net);
    
    network_free(&net);
    return 0;
}


