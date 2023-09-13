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
void matrix_sub(Matrix *out, Matrix *w, Matrix *x);
void matrix_elem_func(Matrix *out, Matrix *m, float (*activation)(float));
void matrix_elem_mul(Matrix *out, Matrix *w, Matrix *x);
void matrix_free(Matrix * m);
void matrix_transpose(Matrix *out, Matrix *m);
void matrix_mse(Matrix *out, Matrix *output, Matrix *label);
void matrix_set_elem(Matrix *m, int row, int col, float val);
void matrix_copy(Matrix *out, Matrix *m);
void matrix_scalar_mul(Matrix *out, float scalar, Matrix *m);

#endif