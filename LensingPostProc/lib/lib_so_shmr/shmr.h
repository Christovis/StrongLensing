typedef struct {
    int *array;
    size_t used;
    size_t size;
} ARRAY;
void initArray(ARRAY *a, size_t initialSize);
void insertArray(ARRAY *a, int element);
void freeArray(ARRAY *a);
void selec_in_rad(float *px1, float *px2, float *px3, float cx1, float cx2, float cx3, int Np, float rad, ARRAY *indx);
float cal_shmr_gal(float *px1, float *px2, float *px3, float cx1, float cx2, float cx3, int Np, float radmax);
