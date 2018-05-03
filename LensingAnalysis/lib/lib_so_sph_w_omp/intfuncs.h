typedef struct{
	float x;
	float y;
	float z;
} PARTICLE;
//----------------------------------------------------------------------
void Loadin_particle_main(long Np,char *fname,PARTICLE *particle);
void Loadin_particle_main_ascii(long Np,char *fname,PARTICLE *particle);
int findHsml(PARTICLE *particle, long *NumP, long * NgbP, double *BS, float * SmoothLength);
void write_3_signals(char *out1o,char *out2o,char *out3o,float *in1,float *in2, float *in3,int Nc, int ind);
float si_weight(float x);
void pin_matrix(long Nc,long i_l,long j_l,long nbx,long nby,float *in1,float *in2,float *out);
//----------------------------------------------------------------------
void Make_cell_SPH_weight(long Nc,float bsz,long Np, PARTICLE *particle, float * SmoothLength, float * mpp, float *sdens);
int cal_sph_sdens_weight(float *x1, float *x2, float *x3,float *mpp,float bsz,long  Nc,float dsx,long Ngb,long Np,float xc1,float xc2,float xc3,float *posx1, float * posx2, float *sdens);
//----------------------------------------------------------------------
void Make_cell_SPH_weight_omp(long Nc,float bsz,long Np, PARTICLE *particle, float * SmoothLength, float *mpp, float *sdens);
int cal_sdens_sp_weight(float x_p,float y_p,float m_p,float hdsl,float dsx,long Nc,float *sdens_sp);
int cal_sph_sdens_weight_omp(float *x1, float *x2, float *x3,float *mpp,float bsz,long  Nc,float dsx,long Ngb,long Np,float xc1,float xc2,float xc3,float *posx1, float * posx2, float *sdens);
