#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <omp.h>
#include "intfuncs.h"
#include "allvars_SPH.h"
#include "proto.h"
//----------------------------------------------------------------------
void Loadin_particle_main(long Np,char *fname,PARTICLE *particle)
{
	long i;
	float x,y,z;
	FILE *f_rb;

	//printf("load in Mainhalo particles from file...");
	f_rb = fopen(fname,"rb");

	for(i=0;i<Np;i++) {
		fread(&x,sizeof(float),1,f_rb);
		fread(&y,sizeof(float),1,f_rb);
		fread(&z,sizeof(float),1,f_rb);

		particle[i].x = (float)(x);
		particle[i].y = (float)(y);
		particle[i].z = (float)(z);
	}
	fclose(f_rb);
	//printf("Loadin finished!\n");
}
//----------------------------------------------------------------------
void Loadin_particle_main_ascii(long Np,char *fname,PARTICLE *particle)
{
  long i;
  FILE *f_r;
  //printf("load in Mainhalo particles from file...");
  f_r = fopen(fname,"r");
  for (i=0; i<Np; i++) {
	  fscanf(f_r, " %f %f %f",&particle[i].x,&particle[i].y,&particle[i].z);
	  //printf("%f %f %f \n",particle[i].x,particle[i].y,particle[i].z);
  }
  fclose(f_r);
  //printf("Loadin finished!\n");
}
//------------------------------------------------------------
void write_3_signals(char *out1o,char *out2o,char *out3o,float *in1,float *in2, float *in3,int Nc, int ind) {
    long i,j,index;
	char out1[256],out2[256],out3[256];
    FILE *f1,*f2,*f3;

	sprintf(out1,"%s_%d.bin",out1o,ind);
	sprintf(out2,"%s_%d.bin",out2o,ind);
	sprintf(out3,"%s_%d.bin",out3o,ind);

    f1 =fopen(out1,"wb");
    f2 =fopen(out2,"wb");
    f3 =fopen(out3,"wb");

    for (i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
        index = i*Nc+j;

        fwrite(&in1[index],sizeof(float),1,f1);
        fwrite(&in2[index],sizeof(float),1,f2);
        fwrite(&in3[index],sizeof(float),1,f3);
    }
    fclose(f1);
    fclose(f2);
    fclose(f3);
}

float si_weight(float x) {
	float sigma = 10.0/(7.0*M_PI);
	float res = 0.0;

	if (x>=0 && x< 1) {
		res = sigma*(1.0-1.5*x*x+0.75*x*x*x);
		return res;
	}
	if (x>=1 && x< 2) {
		res = sigma*0.25*(2.0-x)*(2.0-x)*(2.0-x);
		return res;
	}

	return res;
}

void pin_matrix(long Nc,long i_l,long j_l,long nbx,long nby,float *in1,float *in2,float *out) {
	long i,j,loc_i,loc_j,index1,index2;
	for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
		loc_i = i_l + i;
		loc_j = j_l + j;
		if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

		index1 = loc_i*Nc+loc_j;
		index2 = i*nby+j;
		out[index1] = in1[index1]+in2[index2];
	}
}

//----------------------------------------------------------------------
void Make_cell_SPH_weight(long Nc,float bsz,long Np, PARTICLE *particle, float * SmoothLength, float * mpp, float *sdens) {
	long m,i,j,nbx,nby;//,index;
	float hdsl,R,x_p,y_p;
	long i_l,j_l,i_u,j_u,loc_i,loc_j;
	float dsx = bsz/(float)(Nc);

	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		sdens[i*Nc+j] = 0.0;
	}

	for(m=0;m<Np;m++) {
		if((fabs(particle[m].x) > 0.5*bsz) || (fabs(particle[m].x)> 0.5*bsz)) continue;

		hdsl = SmoothLength[m];

		x_p = particle[m].x+0.5*bsz;
		y_p = particle[m].y+0.5*bsz;

		i_u = (int)((x_p+2.0*hdsl)/dsx);
		i_l = (int)((x_p-2.0*hdsl)/dsx);
		nbx = i_u-i_l+1;

		j_u = (int)((y_p+2.0*hdsl)/dsx);
		j_l = (int)((y_p-2.0*hdsl)/dsx);
		nby = j_u-j_l+1;

		if (nbx == 1 && nby == 1) {
			sdens[i_l*Nc+j_l] += mpp[i]/(dsx*dsx);
			continue;
		}

		for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
			loc_i = i_l + i;
			loc_j = j_l + j;
			if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

			R=sqrt(pow((loc_i+0.5)*dsx-x_p,2)+pow((loc_j+0.5)*dsx-y_p,2));
			sdens[loc_i*Nc+loc_j] +=  mpp[i]*si_weight(R/hdsl)/(hdsl*hdsl);
		}
	}
}

int cal_sph_sdens_weight(float *x1, float *x2, float *x3,float *mpp,float bsz,long  Nc,float dsx,long Ngb,long Np,float xc1,float xc2,float xc3,float *posx1, float * posx2, float *sdens) {

	long i,j,sph;
	PARTICLE *particle = (PARTICLE *)malloc(Np*sizeof(PARTICLE));
	for(i=0;i<Np;i++) {
  	  	particle[i].x = (float)(x1[i]);
  	  	particle[i].y = (float)(x2[i]);
  	  	particle[i].z = (float)(x3[i]);
  	}

	float * SmoothLength = (float *)malloc(Np*sizeof(float));
	double SPHBoxSize = 0.0;
	sph = findHsml(particle,&Np,&Ngb,&SPHBoxSize,SmoothLength);
	if (sph == 1) {
		printf("FindHsml is failed!\n");
	}
	free(particle);

	particle = (PARTICLE *)malloc(Np*sizeof(PARTICLE));
	for(i=0;i<Np;i++) {
  	  	particle[i].x = (float)(x1[i]);
  	  	particle[i].y = (float)(x2[i]);
  	  	particle[i].z = (float)(x3[i]);
  	}
	Make_cell_SPH_weight(Nc,bsz,Np,particle,SmoothLength,mpp,sdens);
	free(SmoothLength);
	free(particle);

	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		posx1[i*Nc+j] = dsx*(float)(i)-bsz*0.5+0.5*dsx+xc1;
		posx2[i*Nc+j] = dsx*(float)(j)-bsz*0.5+0.5*dsx+xc2;
	    sdens[i*Nc+j] = sdens[i*Nc+j];
	}
	return (0);
}

//----------------------------------------------------------------------
void Make_cell_SPH_weight_omp(long Nc,float bsz,long Np, PARTICLE *particle, float * SmoothLength, float *mpp, float *sdens) {
	long m,i,j;
	float hdsl,x_p,y_p,m_p;
	float dsx = bsz/(float)(Nc);

	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		sdens[i*Nc+j] = 0.0;
	}


#pragma omp parallel num_threads(8)	\
	shared(SmoothLength,mpp,sdens,particle,Np,bsz,dsx) \
	private(m,x_p,y_p,m_p,hdsl)
	{
	float *sdens_sp;
	sdens_sp = (float *)calloc(Nc*Nc,sizeof(float));
	#pragma omp for schedule(dynamic,16)
		for(m=0;m<Np;m++) {

			if((fabs(particle[m].x) > 0.5*bsz) || (fabs(particle[m].x)> 0.5*bsz)) continue;

			hdsl = SmoothLength[m];

			x_p = particle[m].x+0.5*bsz;
			y_p = particle[m].y+0.5*bsz;
			m_p = mpp[m];

			cal_sdens_sp_weight(x_p,y_p,m_p,hdsl,dsx,Nc,sdens_sp);
		}
	#pragma omp critical
	{
		for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
			sdens[i*Nc+j] += sdens_sp[i*Nc+j];
		}
	}
	free(sdens_sp);
	}
}

int cal_sdens_sp_weight(float x_p,float y_p,float m_p,float hdsl,float dsx,long Nc,float *sdens_sp) {

	int i_u,i_l,j_u,j_l,nbx,nby;
	int i,j,loc_i,loc_j;
	int Ncr = 32;
	float R;
	float sd_tot;

	i_u = (int)((x_p+2.0*hdsl)/dsx);
	i_l = (int)((x_p-2.0*hdsl)/dsx);
	nbx = i_u-i_l+1;

	j_u = (int)((y_p+2.0*hdsl)/dsx);
	j_l = (int)((y_p-2.0*hdsl)/dsx);
	nby = j_u-j_l+1;


	if (nbx <=2 && nby <=2) {
		sdens_sp[i_l*Nc+j_l] += m_p/(dsx*dsx);
		return 0;
	}

	else {

		if ((nbx+nby)/2 <= Ncr) {
			sd_tot = 0.0;
			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				R=sqrt(pow((loc_i+0.5)*dsx-x_p,2)+pow((loc_j+0.5)*dsx-y_p,2));
				sd_tot += si_weight(R/hdsl)/(hdsl*hdsl)*dsx*dsx;
			}

			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

				R=sqrt(pow((loc_i+0.5)*dsx-x_p,2)+pow((loc_j+0.5)*dsx-y_p,2));
				sdens_sp[loc_i*Nc+loc_j] += m_p*si_weight(R/hdsl)/(hdsl*hdsl)/sd_tot;
			}
			return 0;
		}

		if ((nbx+nby)/2 > 0.0*Ncr) {
			for(i=0;i<nbx;i++) for(j=0;j<nby;j++) {
				loc_i = i_l + i;
				loc_j = j_l + j;
				if((loc_i>=Nc)||(loc_i<0)||(loc_j>=Nc)||(loc_j<0)) continue;

				R=sqrt(pow((loc_i+0.5)*dsx-x_p,2)+pow((loc_j+0.5)*dsx-y_p,2));
				sdens_sp[loc_i*Nc+loc_j] += m_p*si_weight(R/hdsl)/(hdsl*hdsl);
			}
			return 0;
		}
	}
	return 0;
}

int cal_sph_sdens_weight_omp(float *x1, float *x2, float *x3,float *mpp,float bsz,long  Nc,float dsx,long Ngb,long Np,float xc1,float xc2,float xc3,float *posx1, float * posx2, float *sdens) {

	long i,j,sph;
	PARTICLE *particle = (PARTICLE *)malloc(Np*sizeof(PARTICLE));
	for(i=0;i<Np;i++) {
  	  	particle[i].x = (float)(x1[i]);
  	  	particle[i].y = (float)(x2[i]);
  	  	particle[i].z = (float)(x3[i]);
  	}

	float * SmoothLength = (float *)malloc(Np*sizeof(float));
	double SPHBoxSize = 0.0;
	sph = findHsml(particle,&Np,&Ngb,&SPHBoxSize,SmoothLength);
	if (sph == 1) {
		printf("FindHsml is failed!\n");
	}
	free(particle);

	particle = (PARTICLE *)malloc(Np*sizeof(PARTICLE));
	for(i=0;i<Np;i++) {
  	  	particle[i].x = (float)(x1[i]);
  	  	particle[i].y = (float)(x2[i]);
  	  	particle[i].z = (float)(x3[i]);
  	}
	Make_cell_SPH_weight_omp(Nc,bsz,Np,particle,SmoothLength,mpp,sdens);
	free(SmoothLength);
	free(particle);

	for(i=0;i<Nc;i++) for(j=0;j<Nc;j++) {
		posx1[i*Nc+j] = dsx*(float)(i)-bsz*0.5+0.5*dsx+xc1;
		posx2[i*Nc+j] = dsx*(float)(j)-bsz*0.5+0.5*dsx+xc2;
	    sdens[i*Nc+j] = sdens[i*Nc+j];
	}
	return (0);
}
