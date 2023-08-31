#ifndef MATRIX_H
#define MATRIX_H

#define MAT_IDX(m,i,j) (m)->data[(i)*(m)->cols + (j)]

typedef struct Matrix {
    int rows;
    int cols;
    float *data;
} Matrix;

Matrix matrix_alloc(int rows, int cols);
void matrix_mul(Matrix *out, Matrix *w, Matrix *x);
void matrix_rand(Matrix *out, float low, float high);
void matrix_print(Matrix *m);
void matrix_add(Matrix *out, Matrix *w, Matrix *x);
void matrix_act_func(Matrix *out, Matrix *m, float (*activation)(float));
void matrix_elem_mul(Matrix *out, Matrix *w, Matrix *x);

#endif