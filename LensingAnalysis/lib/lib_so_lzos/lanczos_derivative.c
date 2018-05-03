void lanczos_diff_1_tag(double *mi, double *m1, double *m2, double Dcell, int Ncc, int dif_tag) {
	int i_m3,i_p3,j_m3,j_p3,i_m2,i_p2,j_m2,j_p2,i_m1,j_m1,i_p1,j_p1,i,j;
	int index;

	for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {
		if (i==0) {i_m1 = Ncc-1;i_m2 = Ncc-2;i_m3 = Ncc-3;}
		else if (i==1) {i_m1 = 0;i_m2 = Ncc-1;i_m3 = Ncc-2;}
		else if (i==2) {i_m1 = 1;i_m2 = 0;i_m3 = Ncc-1;}
		else {i_m1 = i-1;i_m2 = i-2;i_m3 = i-3;}
		if (j==0) {j_m1 = Ncc-1;j_m2 = Ncc-2;j_m3 = Ncc-3;}
		else if (j==1) {j_m1 = 0;j_m2 = Ncc-1;j_m3 = Ncc-2;}
		else if (j==2) {j_m1 = 1;j_m2 = 0;j_m3 = Ncc-1;}
		else {j_m1 = j-1;j_m2 = j-2;j_m3 = j-3;}
		if (i==Ncc-1) {i_p1 = 0;i_p2 = 1;i_p3 = 2;}
		else if (i==Ncc-2) {i_p1 = Ncc-1;i_p2 = 0;i_p3 = 1;}
		else if (i==Ncc-3) {i_p1 = Ncc-2;i_p2 = Ncc-1;i_p3 = 0;}
		else {i_p1 = i+1;i_p2 = i+2;i_p3 = i+3;}
		if (j==Ncc-1) {j_p1 = 0;j_p2 = 1;j_p3 = 2;}
		else if (j==Ncc-2) {j_p1 = Ncc-1;j_p2 = 0;j_p3 = 1;}
		else if (j==Ncc-2) {j_p1 = Ncc-2;j_p2 = Ncc-1;j_p3 = 0;}
		else {j_p1 = j+1;j_p2 = j+2;j_p3 = j+3;}

		index = i*Ncc+j;


		if (dif_tag==0) {
			m1[index] = (mi[i_p1*Ncc+j]-mi[i_m1*Ncc+j])*2.0/3.0/Dcell
			    	  - (mi[i_p2*Ncc+j]-mi[i_m2*Ncc+j])/12.0/Dcell;
			m2[index] = (mi[i*Ncc+j_p1]-mi[i*Ncc+j_m1])*2.0/3.0/Dcell
					  - (mi[i*Ncc+j_p2]-mi[i*Ncc+j_m2])/12.0/Dcell;
		}

		if (dif_tag==1) {
			m1[index] =(1.0*(mi[i_p1*Ncc+j]-mi[i_m1*Ncc+j])
			  		  + 2.0*(mi[i_p2*Ncc+j]-mi[i_m2*Ncc+j])
			   		  + 3.0*(mi[i_p3*Ncc+j]-mi[i_m3*Ncc+j]))/(28.0*Dcell);
			m2[index] =(1.0*(mi[i*Ncc+j_p1]-mi[i*Ncc+j_m1])
			  		  + 2.0*(mi[i*Ncc+j_p2]-mi[i*Ncc+j_m2])
			   		  + 3.0*(mi[i*Ncc+j_p3]-mi[i*Ncc+j_m3]))/(28.0*Dcell);
		}

		if (dif_tag==2) {
			m1[index] =(5.0*(mi[i_p1*Ncc+j]-mi[i_m1*Ncc+j])
			  	   	  + 4.0*(mi[i_p2*Ncc+j]-mi[i_m2*Ncc+j])
			     	  + 1.0*(mi[i_p3*Ncc+j]-mi[i_m3*Ncc+j]))/(32.0*Dcell);
			m2[index] =(5.0*(mi[i*Ncc+j_p1]-mi[i*Ncc+j_m1])
			  		  + 4.0*(mi[i*Ncc+j_p2]-mi[i*Ncc+j_m2])
			  		  + 1.0*(mi[i*Ncc+j_p3]-mi[i*Ncc+j_m3]))/(32.0*Dcell);
		}

		if (dif_tag==3) {
			m1[index] = (58.0*(mi[i_p1*Ncc+j]-mi[i_m1*Ncc+j])
			            + 67.0*(mi[i_p2*Ncc+j]-mi[i_m2*Ncc+j])
			  	   	 	+ 22.0*(mi[i_p3*Ncc+j]-mi[i_m3*Ncc+j]))/(252.0*Dcell);
			m2[index] = (58.0*(mi[i*Ncc+j_p1]-mi[i*Ncc+j_m1])
					    + 67.0*(mi[i*Ncc+j_p2]-mi[i*Ncc+j_m2])
						- 22.0*(mi[i*Ncc+j_p3]-mi[i*Ncc+j_m3]))/(252.0*Dcell);
		}
	}
}
//--------------------------------------------------------------------
void lanczos_diff_2_tag(double *m1, double *m2, double *m11, double *m12, double *m21, double *m22, double Dcell, int Ncc, int dif_tag) {
	int i_m3,i_p3,j_m3,j_p3,i_m2,i_p2,j_m2,j_p2,i_m1,j_m1,i_p1,j_p1,i,j;
	int index;

	for(i=0;i<Ncc;i++) for(j=0;j<Ncc;j++) {
		if (i==0) {i_m1 = Ncc-1;i_m2 = Ncc-2;i_m3 = Ncc-3;}
		else if (i==1) {i_m1 = 0;i_m2 = Ncc-1;i_m3 = Ncc-2;}
		else if (i==2) {i_m1 = 1;i_m2 = 0;i_m3 = Ncc-1;}
		else {i_m1 = i-1;i_m2 = i-2;i_m3 = i-3;}
		if (j==0) {j_m1 = Ncc-1;j_m2 = Ncc-2;j_m3 = Ncc-3;}
		else if (j==1) {j_m1 = 0;j_m2 = Ncc-1;j_m3 = Ncc-2;}
		else if (j==2) {j_m1 = 1;j_m2 = 0;j_m3 = Ncc-1;}
		else {j_m1 = j-1;j_m2 = j-2;j_m3 = j-3;}
		if (i==Ncc-1) {i_p1 = 0;i_p2 = 1;i_p3 = 2;}
		else if (i==Ncc-2) {i_p1 = Ncc-1;i_p2 = 0;i_p3 = 1;}
		else if (i==Ncc-3) {i_p1 = Ncc-2;i_p2 = Ncc-1;i_p3 = 0;}
		else {i_p1 = i+1;i_p2 = i+2;i_p3 = i+3;}
		if (j==Ncc-1) {j_p1 = 0;j_p2 = 1;j_p3 = 2;}
		else if (j==Ncc-2) {j_p1 = Ncc-1;j_p2 = 0;j_p3 = 1;}
		else if (j==Ncc-2) {j_p1 = Ncc-2;j_p2 = Ncc-1;j_p3 = 0;}
		else {j_p1 = j+1;j_p2 = j+2;j_p3 = j+3;}

		index = i*Ncc+j;

		if (dif_tag==0) {
			m11[index] = (m1[i_p1*Ncc+j]-m1[i_m1*Ncc+j])*2.0/3.0/Dcell
			  		   - (m1[i_p2*Ncc+j]-m1[i_m2*Ncc+j])/12.0/Dcell;
			m22[index] = (m2[i*Ncc+j_p1]-m2[i*Ncc+j_m1])*2.0/3.0/Dcell
			  		   - (m2[i*Ncc+j_p2]-m2[i*Ncc+j_m2])/12.0/Dcell;
			m12[index] = (m2[i_p1*Ncc+j]-m2[i_m1*Ncc+j])*2.0/3.0/Dcell
			  		   - (m2[i_p2*Ncc+j]-m2[i_m2*Ncc+j])/12.0/Dcell;
			m21[index] = (m1[i*Ncc+j_p1]-m1[i*Ncc+j_m1])*2.0/3.0/Dcell
					   - (m1[i*Ncc+j_p2]-m1[i*Ncc+j_m2])/12.0/Dcell;
		}

		if (dif_tag==1) {
			m11[index] =(1.0*(m1[i_p1*Ncc+j]-m1[i_m1*Ncc+j])
			  		   + 2.0*(m1[i_p2*Ncc+j]-m1[i_m2*Ncc+j])
			   		   + 3.0*(m1[i_p3*Ncc+j]-m1[i_m3*Ncc+j]))/(28.0*Dcell);
			m22[index] =(1.0*(m2[i*Ncc+j_p1]-m2[i*Ncc+j_m1])
			  		   + 2.0*(m2[i*Ncc+j_p2]-m2[i*Ncc+j_m2])
			   		   + 3.0*(m2[i*Ncc+j_p3]-m2[i*Ncc+j_m3]))/(28.0*Dcell);
			m12[index] =(1.0*(m1[i*Ncc+j_p1]-m1[i*Ncc+j_m1])
			  		   + 2.0*(m1[i*Ncc+j_p2]-m1[i*Ncc+j_m2])
			   		   + 3.0*(m1[i*Ncc+j_p3]-m1[i*Ncc+j_m3]))/(28.0*Dcell);
			m21[index] =(1.0*(m2[i_p1*Ncc+j]-m2[i_m1*Ncc+j])
					   + 2.0*(m2[i_p2*Ncc+j]-m2[i_m2*Ncc+j])
			     	   + 3.0*(m2[i_p3*Ncc+j]-m2[i_m3*Ncc+j]))/(28.0*Dcell);
		}

		if (dif_tag==2) {
			m11[index] = (5.0*(m1[i_p1*Ncc+j]-m1[i_m1*Ncc+j])
			  	   	    + 4.0*(m1[i_p2*Ncc+j]-m1[i_m2*Ncc+j])
			     	 	+ 1.0*(m1[i_p3*Ncc+j]-m1[i_m3*Ncc+j]))/(32.0*Dcell);
			m22[index] = (5.0*(m2[i*Ncc+j_p1]-m2[i*Ncc+j_m1])
			  		    + 4.0*(m2[i*Ncc+j_p2]-m2[i*Ncc+j_m2])
			  		    + 1.0*(m2[i*Ncc+j_p3]-m2[i*Ncc+j_m3]))/(32.0*Dcell);
			m12[index] = (5.0*(m1[i*Ncc+j_p1]-m1[i*Ncc+j_m1])
			  		    + 4.0*(m1[i*Ncc+j_p2]-m1[i*Ncc+j_m2])
			  		    + 1.0*(m1[i*Ncc+j_p3]-m1[i*Ncc+j_m3]))/(32.0*Dcell);
			m21[index] = (5.0*(m2[i_p1*Ncc+j]-m2[i_m1*Ncc+j])
					    + 4.0*(m2[i_p2*Ncc+j]-m2[i_m2*Ncc+j])
					 	+ 1.0*(m2[i_p3*Ncc+j]-m2[i_m3*Ncc+j]))/(32.0*Dcell);
		}
	}
}
