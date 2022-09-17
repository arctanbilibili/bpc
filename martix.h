#ifndef _MARTIX_H_
#define _MARTIX_H_

#include "malloc.h"
#include <stdio.h>

#define NONE "\033[m"
#define RED "\033[0;32;31m"
#define LIGHT_RED "\033[1;31m"
#define GREEN "\033[0;32;32m"
#define LIGHT_GREEN "\033[1;32m"
#define BLUE "\033[0;32;34m"
#define LIGHT_BLUE "\033[1;34m"
#define DARY_GRAY "\033[1;30m"
#define CYAN "\033[0;36m"
#define LIGHT_CYAN "\033[1;36m"
#define PURPLE "\033[0;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN "\033[0;33m"
#define YELLOW "\033[1;33m"
#define LIGHT_GRAY "\033[0;37m"
#define WHITE "\033[1;37m"

typedef struct{
    int rol;
    int col;
    double *val;
}MARTIX;

MARTIX* MARTIX_cre(int rol,int col){
    MARTIX* mat = (MARTIX*)malloc(sizeof(MARTIX));
    mat->rol = rol;
    mat->col = col;
    mat->val = (double*)malloc(rol*col*sizeof(double));
    return mat;
}

void MARTIX_del(MARTIX* mat){
    free(mat->val);
    free(mat);
}

MARTIX* MARTIX_copy(MARTIX*mat0){
    // MARTIX* mat = (MARTIX*)malloc(sizeof(MARTIX));
    // mat->rol = mat0->rol;
    // mat->col = mat0->col;
    // mat->val = (double*)malloc(mat0->rol*mat0->col*sizeof(double));
    MARTIX* mat = MARTIX_cre(mat0->rol,mat0->col);

    int size0 = mat0->rol*mat0->col;
    for(int i=0;i<size0;i++){
        mat->val[i] = mat0->val[i];
    }

    return mat;
}

void MARTIX_ncover(MARTIX*A,MARTIX*B){//A<=B//不会删除原来的B
    A->rol = B->rol;
    A->col = B->col;
    int len = A->rol*A->col;
    for(int i=0;i<len;i++){
        A->val[i] = B->val[i];
    }
}

void MARTIX_cover(MARTIX*A,MARTIX*B){//A<=B//会删除原来的B
    A->rol = B->rol;
    A->col = B->col;
    int len = A->rol*A->col;
    for(int i=0;i<len;i++){
        A->val[i] = B->val[i];
    }
    MARTIX_del(B);
}

MARTIX* MARTIX_mul(MARTIX*A,MARTIX*B){//返回值会额外产生一个副本，配合cover使用
    int rol = A->rol;
    int col = B->col;
    MARTIX* mul = MARTIX_cre(rol,col);

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            double sum=0;
            for(int i=0;i<A->col;i++){
                sum += A->val[r*A->col + i] * B->val[i*B->col + c];
            }
            mul->val[col*r+c] = sum;
        }
    }
    return mul;
}

void MARTIX_mul_dot_cover(MARTIX*dst,MARTIX*A,MARTIX*B){//直接覆盖dst=A.*B
    int rol = A->rol;
    int col = A->col;

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            dst->val[r*col+c] = A->val[r*col+c] * B->val[r*col+c];
        }
    }
}

void MARTIX_mul_cover(MARTIX*dst,MARTIX*A,MARTIX*B){//不会产生额外冗余内存
    MARTIX* tmp = MARTIX_mul(A,B);
    MARTIX_cover(dst,tmp);
}

MARTIX* MARTIX_add(MARTIX*A,MARTIX*B){//A需要自己del
    int rol = A->rol;
    int col = A->col;
    MARTIX* sum = MARTIX_cre(rol,col);
    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            sum->val[r*col+c] = A->val[r*col+c] + B->val[r*col+c];
        }
    }
    return sum;
}

void MARTIX_add_cover(MARTIX*dst,MARTIX*A,MARTIX*B){//不需要自己del
    int rol = A->rol;
    int col = A->col;

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            dst->val[r*col+c] = A->val[r*col+c] + B->val[r*col+c];
        }
    }
}

MARTIX* MARTIX_sub(MARTIX*A,MARTIX*B){//A需要自己del
    int rol = A->rol;
    int col = A->col;
    MARTIX* sum = MARTIX_cre(rol,col);
    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            sum->val[r*col+c] = A->val[r*col+c] - B->val[r*col+c];
        }
    }
    return sum;
}
void MARTIX_sub_cover(MARTIX*dst,MARTIX*A,MARTIX*B){//直接覆盖dst=A-B
    int rol = A->rol;
    int col = A->col;

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            dst->val[r*col+c] = A->val[r*col+c] - B->val[r*col+c];
        }
    }
}

void MARTIX_add_para_cover(MARTIX*dst,MARTIX*A,double s){//直接覆盖dst=A+s
    int rol = A->rol;
    int col = A->col;

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            dst->val[r*col+c] = A->val[r*col+c]+s;
        }
    }
}
void MARTIX_mul_para_cover(MARTIX*dst,MARTIX*A,double s){//直接覆盖dst=A*s
    int rol = A->rol;
    int col = A->col;

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            dst->val[r*col+c] = A->val[r*col+c]*s;
        }
    }
}

MARTIX* MARTIX_T(MARTIX*A){//需要del
    int rol = A->rol;
    int col = A->col;
    MARTIX* T = MARTIX_cre(col,rol);

    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            T->val[c*rol + r] = A->val[r*col + c];
        }
    }
    return T;
}

MARTIX* MARTIX_cre_paras(int rol,int col,double s){//创建全同常数矩阵 需要自己del
    MARTIX* mat = MARTIX_cre(rol,col);

    int size0 = rol*col;
    for(int i=0;i<size0;i++){
        mat->val[i] = s;
    }

    return mat;
}

double MARTIX_sum(MARTIX*A){
    double sum=0;
    int rol = A->rol;
    int col = A->col;
    for(int r=0;r<rol;r++){
        for(int c=0;c<col;c++){
            sum += A->val[r*col+c];
        }
    }
    return sum;
}

//会覆盖B
void MARTIX_f(MARTIX*B,MARTIX*A,double(*f)(double)){
    for(int rol=0;rol<A->rol;rol++){
        for(int col=0;col<A->col;col++){
            B->val[rol*A->col+col] = f(A->val[rol*A->col+col]);
        }
    }
}


void showMat(MARTIX*mat,const char * name){
    if(name!=0)
        printf(RED"%s\r\n"NONE,name);
    if(mat==NULL){
        printf(RED"NULL\r\n"NONE);
    }
    else{
        for(int r=0;r<mat->rol;r++){
            for(int c=0;c<mat->col;c++){
                printf("%11.8lf ",mat->val[r*mat->col+c]);
            }
            printf("\r\n");
        }
    }
}

void showMat_f(MARTIX*mat,const char * name,FILE* fp){
    if(name!=0)
        fprintf(fp,"%s\n",name);
    if(mat==NULL){
        fprintf(fp,"NULL\n");
    }
    else{
        for(int r=0;r<mat->rol;r++){
            for(int c=0;c<mat->col;c++){
                fprintf(fp,"%11.8lf ",mat->val[r*mat->col+c]);
            }
            fprintf(fp,"\n");
        }
    }
}

#endif