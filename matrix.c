#include "matrix.h"
#include <stdlib.h>
#include <math.h>
#include "utils.h"
#include <assert.h>
#include <stdio.h>

Matrix matrix_alloc(int rows, int cols)
{
    //Matrix *m = malloc(sizeof(Matrix));
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(rows*cols*sizeof(*(m.data)));
    assert(m.data != NULL);
   return m;
}

void matrix_act_func(Matrix *out, Matrix *m, float (*activation)(float))
{
    assert(out->cols == m->cols);
    assert(out->rows == m->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = (*activation)(MAT_IDX(m,i,j));
        }
    }
}

void matrix_add(Matrix *out, Matrix *w, Matrix *x)
{
    assert(w->cols==x->cols);
    assert(w->rows==x->cols);
    assert(out->cols==w->cols);
    assert(out->rows==w->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = MAT_IDX(w,i,j) + MAT_IDX(x,i,j);
        }
    }
}

void matrix_mul(Matrix *out, Matrix *w, Matrix *x) 
{
    // W (d,n) @ x (n,p) -> out (d,p)
    assert(out->cols == x->cols);
    assert(out->rows == w->rows);
    assert(w->cols == x->rows);
    int n = w->cols;
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = 0;
            for (int k = 0; k < n; ++k) {
                MAT_IDX(out,i,j) += MAT_IDX(w,i,k) * MAT_IDX(x,k,j);
            }
        }
    }
}

void matrix_rand(Matrix *out, float low, float high)
{
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = frand()*(high-low)+low;
        }
    }
}

void matrix_print(Matrix *m)
{
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            printf("%f ",MAT_IDX(m,i,j));
        }
        printf("\n");
    }
}
