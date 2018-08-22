#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

// Dynamical array
typedef struct {
    int *array;
    size_t used;
    size_t size;
} ARRAY;

void initArray(ARRAY *a, size_t initialSize) {
    a->array = (int * ) malloc(initialSize * sizeof(int));
    a->used = 0;
    a->size = initialSize;
}

void insertArray(ARRAY *a, int element) {
    if (a->used == a->size) {
        a->size *= 2;
        a->array = (int * )realloc(a->array, a->size * sizeof(int));
      }
      a->array[a->used++] = element;
}

void freeArray(ARRAY *a) {
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

// Save indices of particles in sphere
void selec_in_rad(float *px1, float *px2, float *px3, float cx1, float cx2,
                  float cx3, int Np, float rad, ARRAY *indx) {
    int i;
    float dist;

    for(i=0; i<Np; i++) {
        dist = pow((px1[i] - cx1), 2) +\
               pow((px2[i] - cx2), 2) +\
               pow((px3[i] - cx3), 2);
        if(dist<rad) {
            insertArray(indx, i);
        }
    }
}


void bisection (float *rad, float a, float b, int *itr) {
    *rad=(a+b)/2;
    ++(*itr);
}


float mass_in_rad(float *px1, float *px2, float *px3, float cx1, float cx2, float cx3,
                  float *pmass, int Np, float radmax) {
//float mass_in_rad(float *pmass, ARRAY *indx) {
    int i;
    float masstot=0.0;
    ARRAY indx;
    initArray(&indx, 5);

    selec_in_rad(px1, px2, px3, cx1, cx2, cx3, Np, radmax, &indx);
    for(i=0; i<indx.used; i++) {
        masstot += pmass[indx.array[i]];
    }
    return masstot;
}


float cal_shmr_gal(float *px1, float *px2, float *px3, float cx1, float cx2, float cx3,
                   float *pmass, int Np, float radmax) {
    int itr=0, \
        maxmitr=200;
    float pmassmax, \
          pmasshalf, \
          pmass_a, \
          pmass_b, \
          rad, \
          a=0.0, \
          b=radmax, \
          allerr=0.0001, \
          x1;
    //ARRAY indx;
    //initArray(&indx, 5);

    //selec_in_rad(px1, px2, px3, cx1, cx2, cx3, Np, radmax, &indx);
    pmassmax = mass_in_rad(px1, px2, px3, cx1, cx2, cx3, pmass, Np, radmax);
    //pmassmax = mass_in_rad(pmass, &indx);

    bisection(&rad, a, b, &itr);
    do {
        pmass_a = mass_in_rad(px1, px2, px3, cx1, cx2, cx3, pmass, Np, a);
        //selec_in_rad(px1, px2, px3, cx1, cx2, cx3, Np, a, &indx);
        //pmassmax = mass_in_rad(pmass, &indx);
        pmass_a -= pmassmax*0.5;
        pmass_b = mass_in_rad(px1, px2, px3, cx1, cx2, cx3, pmass, Np, rad);
        //selec_in_rad(px1, px2, px3, cx1, cx2, cx3, Np, rad, &indx);
        //pmassmax = mass_in_rad(pmass, &indx);
        pmass_b -= pmassmax*0.5;
        if (pmass_a*pmass_b < 0)
            b = rad;
        else
            a = rad;
        bisection(&x1, a, b, &itr);
        pmasshalf = mass_in_rad(px1, px2, px3, cx1, cx2, cx3, pmass, Np, x1);
        if (fabs(pmasshalf/pmassmax - 0.5) < allerr) {
            //printf("Stellar Half Mass Radius: %f \n", pmasshalf/pow(10, 10));
            return x1;
        }
        rad=x1;
    } while (itr < maxmitr);
    
    printf("ERROR SHMR: iterations are not sufficient. \n");
    return 0;
}
