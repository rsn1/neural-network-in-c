#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include "utils.h"
#include "matrix.h"

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

//elementwise apply func
void matrix_elem_func(Matrix *out, Matrix *m, float (*func)(float))
{
    assert(out->cols == m->cols);
    assert(out->rows == m->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = (*func)(MAT_IDX(m,i,j));
        }
    }
}

void matrix_add(Matrix *out, Matrix *w, Matrix *x)
{
    assert(w->cols==x->cols);
    assert(w->rows==x->rows);
    assert(out->cols==w->cols);
    assert(out->rows==w->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = MAT_IDX(w,i,j) + MAT_IDX(x,i,j);
        }
    }
}

void matrix_set_elem(Matrix *m, int row, int col, float val)
{
    MAT_IDX(m,row,col) = val;
}

//w-x
void matrix_sub(Matrix *out, Matrix *w, Matrix *x)
{
    assert(w->cols==x->cols);
    assert(w->rows==x->rows);
    assert(out->cols==w->cols);
    assert(out->rows==w->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = MAT_IDX(w,i,j) - MAT_IDX(x,i,j);
        }
    }
}

void matrix_mul(Matrix *out, Matrix *w, Matrix *x) 
{
    // w (d,n) @ x (n,p) -> out (d,p)
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

void matrix_elem_mul(Matrix *out, Matrix *w, Matrix *x) 
{
    // w (d,n) * x (d,n) -> out (d,n)
    assert(w->cols==x->cols);
    assert(w->rows==x->rows);
    assert(out->cols==w->cols);
    assert(out->rows==w->rows);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = MAT_IDX(w,i,j) * MAT_IDX(x,i,j);
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

void matrix_transpose(Matrix *out, Matrix *m)
{
    Matrix m_t = matrix_alloc(m->cols,m->rows);
    for (int i = 0; i < m->rows; ++i) {
        for (int j = 0; j < m->cols; ++j) {
            MAT_IDX(&m_t,j,i) = MAT_IDX(m,i,j);
        }
    }
}

void mse(Matrix *out, Matrix *a, Matrix *b)
{
	assert(a->cols == b->cols);
	assert(a->rows == b->rows);
    assert(out->cols == b->cols);
    assert(out->rows == b->rows);
	matrix_sub(out,a,b);
    for (int i = 0; i < out->rows; ++i) {
        for (int j = 0; j < out->cols; ++j) {
            MAT_IDX(out,i,j) = powf(MAT_IDX(out,i,j),2);
        }
    }
}

void mse_d(Matrix *a, Matrix *b) 
{

	
}


void matrix_free(Matrix *m)
{
    free(m->data);
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
