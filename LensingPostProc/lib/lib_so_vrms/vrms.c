#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>


float cal_std(double data[], long Np) {
    float sum = 0.0, mean, standardDeviation = 0.0;
    int i;

    for(i=0; i<Np; ++i) {
        sum += data[i];
    }

    mean = sum/Np;

    for(i=0; i<Np; ++i)
        standardDeviation += pow(data[i] - mean, 2);

    return sqrt(standardDeviation/Np);
}


float median(int n, int x[]) {
    float temp;
    int i, j;
    for(i=0; i<n-1; i++) {
        for(j=i+1; j<n; j++) {
            if(x[j] < x[i]) {
                // swap elements
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }
    }
    if(n%2==0) {
        // if even number of elements, return mean of the two elements in the middle
        return((x[n/2] + x[n/2 - 1]) / 2.0);
    } else {
        // else return the element in the middle
        return x[n/2];
    }
}


double vector_magnitude(double *vec) {
    double sums=0.0, magnitude=0.0;

    sums = pow(vec[0], 2) + pow(vec[1], 2) + pow(vec[2], 2);
    magnitude = sqrt((double)sums);
    return magnitude;
}


float cal_vrms_gal(float *pv1, float *pv2, float *pv3, float gv1, float gv2, float gv3,
                   float *sl1, float *sl2, float *sl3, int Np, int Ns, float sigma) {
    int i, j;
    double *pv_proj = (double *) malloc(3 * sizeof(double));
    double *pv_norm = (double *) malloc(Np * sizeof(double));

    // particle velocity relative to subhalo velocity
    for(i=0; i<Np; i++) {
        pv1[i] += gv1;
        pv2[i] += gv2;
        pv3[i] += gv3;
    }
    
    for(i=0; i<Ns; i++) {
        for(j=0; j<Np; j++) {
            // project particle velocities on slices
            pv_proj[0] = pv1[j]*sl1[i];
            pv_proj[1] = pv2[j]*sl2[i];
            pv_proj[2] = pv3[j]*sl3[i];
            pv_norm[j] = vector_magnitude(pv_proj);
        }
        // add standard deviation in slice
        sigma += pow(cal_std(pv_norm, Np), 2);
    } 

    sigma = sqrt(sigma/Ns);
    return sigma;
}
