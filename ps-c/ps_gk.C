#include <math.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <fftw3.h> 
//#include <gsl/gsl_sf_trig.h>

FILE *fp; 

float kn, kf, coeff; 

/*
double fourier_space_grid_function(double k1, double k2, double k3, int nbox) {

	double w; 
	double k1arg, k2arg, k3arg; 
    
	k1arg = k1/(2.0*(double)kn); 
	k2arg = k2/(2.0*(double)kn);
	k3arg = k3/(2.0*(double)kn);

	w = gsl_sf_sinc(k1arg)*gsl_sf_sinc(k2arg)*gsl_sf_sinc(k3arg); 
	w = w*w; 

	return w;
    
}
*/

void myps(int nbox, float *grid, float box_size, char *filename) {

	fftw_complex *in, *out; 
	fftw_plan p; 
	long long N;
	int i, j, k, sign, nhalf, iw, i2, i1, i3, nkweights; 
	unsigned flags;
	float g ; 
	double *w, *deltasqk, *powspec; 
	long long *iweights;				
	double tpisq;
	float *local_grid; 
	long int N_gridpoints; 
	double shot_noise_correction, c1; 
	double error, tolerance, k1, k2, p1, p2, alpha0, *powspec_dummy, *c2, g_local; 
	int i_dummy, i_c2, ii, jj, kk, i_local, j_local, k_local;
	int counter; 
	double old_alpha0, contrib, sum;
	float *transform;
	long long ilong, jlong, klong, nbox_long, ilong1, ilong2, ilong3, m;
	int status;

	local_grid = grid;

	N = nbox;
	N *= nbox; 
	N *= nbox;

	printf("myps: N=%lld\n", N);

	nbox_long = (long long) nbox;

	printf("myps: nbox=%d\n", nbox);
	printf("myps: nbox_long=%lld\n", nbox_long);
	printf("myps: nbox_long=%lld\n", nbox_long);
	printf("myps: box_size=%f\n", box_size);

	in = (fftw_complex *) fftw_malloc(sizeof(fftw_complex)*N); 
	out = in; 

	status = fftw_import_wisdom_from_filename("wisdom.dat");
	if (status == 0)
		printf("myps: problem reading wisdom\n");
	
	p = fftw_plan_dft_3d(nbox, nbox, nbox, in, out, FFTW_FORWARD, FFTW_MEASURE);

	/* status = fftw_export_wisdom_to_filename("wisdom.dat"); */
	/* if (status == 0) */
	/* 	printf("myps: problem writing wisdom\n"); */
	
	// ----------------------

	kf = 2*M_PI/box_size; // h/Mpc 
	kn = M_PI*(float)nbox/box_size; // h/Mpc; Nyquist wavenumber
	coeff = pow(box_size/(2.0*M_PI),2.0); // Mpc/h
	printf("myps: kf=%f, kn=%f\n", kf, kn);

	nhalf = nbox / 2;
	w = (double *) malloc(nbox*sizeof(double));
	for(i=0; i<nbox; i++) {
		if (i > nhalf) 
			iw = i-nbox; 
		else 
			iw = i; 
		w[i] = kf*(double)iw;
		printf("%e ", w[i]);
	}
	printf("\n");
	// ----------------------

	for(ilong=0; ilong<N; ilong++) {

		if(ilong < 10)
			printf("local_grid: %lld  %e\n", ilong, local_grid[ilong]);
		
		in[ilong][0] = local_grid[ilong];
		in[ilong][1] = 0.0;
	}
	for (ilong=0; ilong<2; ilong++) 
		for (jlong=0; jlong<2; jlong++) 
			for (klong=0; klong<2; klong++) {
				printf("in: %lld %lld %lld %f %f \n", 
					ilong, jlong, klong, in[klong+nbox_long*(jlong+nbox_long*ilong)][0], 
					in[klong+nbox_long*(jlong+nbox_long*ilong)][1]);
			}
		
	fftw_execute(p); 

	// ---------------------
  
	contrib = 0.0;
	powspec=(double *) calloc(nbox_long, sizeof(double));
	for (ilong=0; ilong<nbox_long; ilong++) 
		for (jlong=0; jlong<nbox_long; jlong++) 
			for (klong=0; klong<nbox_long; klong++) {

				g = w[ilong]*w[ilong] + w[jlong]*w[jlong] + w[klong]*w[klong]; 

				if (g != 0) {
					i1 = (int)(0.5+sqrt(g*coeff));
					contrib = pow(out[klong+nbox_long*(jlong+nbox_long*ilong)][0],2.0) + 
						pow(out[klong+nbox_long*(jlong+nbox_long*ilong)][1],2.0);
					//contrib /= pow(fourier_space_grid_function(w[ilong],w[j],w[k],nbox),2.0);
					powspec[i1] += contrib; 
					if(ilong==1 && jlong==1) {
						printf("Contrib: %d %d %d %f %f %d %f %fj %e \n", 
						ilong, jlong, klong, g, coeff, i1, out[klong+nbox_long*(jlong+nbox_long*ilong)][0], 
						out[klong+nbox_long*(jlong+nbox_long*ilong)][1], contrib); 
					}
				}
			}

#ifdef GET_KSPACE_SLICE
	fp = fopen("fourier_slice.dat", "w");
	ilong = 0;
	jlong = 0;
	for (klong=0; klong<nbox_long; klong++) 
		fprintf(fp, "%lld  %lld  %lld  %e\n", ilong, jlong, klong, out[klong+nbox_long*(jlong+nbox_long*ilong)][0]);
	fclose(fp);
#endif
    
	fftw_free(in); 
	
	for (i = 0; i < nhalf; i++) {
		printf("%f  %e \n", w[i], powspec[i]); 
	}
	printf("Power calculation done\n");

	// ----------------------

	iweights=(long long *) malloc(nbox_long*sizeof(long long));
	for(ilong=0;ilong<nbox_long;ilong++)
		iweights[ilong]=0;
	tpisq=2.0*M_PI*M_PI;

	for(ilong=0;ilong<nbox_long;ilong++){
		ilong1=ilong;
		if(ilong1>=nhalf)
			ilong1=nbox_long-ilong1;
		for(jlong=0;jlong<nbox_long;jlong++){
			ilong2=jlong;
			if(ilong2>=nhalf)
				ilong2=nbox_long-ilong2;
			for(klong=0;klong<nbox_long;klong++){
				ilong3=klong;
				if(ilong3>=nhalf)
					ilong3=nbox_long-ilong3;
				m=0.5+sqrt(ilong1*ilong1+ilong2*ilong2+ilong3*ilong3);
				iweights[m]+=1;
			}//for...klong
		}//for...jlong
	}//for...i

	// ----------------------

	deltasqk = (double *) calloc(nbox_long, sizeof(double));

	for (i = 0; i < nhalf; i++) {
		powspec[i] = powspec[i]*pow(box_size,3.0)/pow((float)nbox,6.0);
		powspec[i] /= (double)iweights[i];
		deltasqk[i] = pow(w[i],3.0)*powspec[i]/(2.0*M_PI*M_PI);
	}
  
	// ----------------------

	/* Correct shot noise effect.  See Equations 19 and 21 of Jing (2005
	 * ApJ 620 559). */

	//N_gridpoints = nbox*nbox*nbox; 
	/* for (i = 0; i < nhalf; i++) { */
	/* 	c1 = 1.0 - (2.0/3.0)*pow(sin(M_PI*w[i]/(2*kn)),2); */
	/* 	shot_noise_correction = c1/((double)N); */
	/* 	powspec[i] -= shot_noise_correction;  */
	/* 	deltasqk[i] = pow(w[i],3.0)*powspec[i]/(2.0*M_PI*M_PI); */
	/* } */

	// ----------------------
  

	// ----------------------

	fp = fopen(filename, "w"); 
	for (i = 0; i < nhalf; i++) 
		fprintf(fp, "%f  %e  %e  %lld\n", w[i], powspec[i], deltasqk[i], iweights[i]); 
	fclose(fp);

	free(iweights); 

	
	/* sum = 0.0; */
	/* for (i = 0; i < nhalf-1; i++)  */
	/* 	sum += (w[i+1]-w[i])*deltasqk[i]/k; */

	/* printf("Variance from power spectrum: %e\n",sum); */

	fftw_destroy_plan(p); 
	free(w); 
  
	return; 
  
}

float *read_data(char *filename) {
	FILE *fp;
	fp = fopen(filename, "r");
	int nx, ny, nz;
	fscanf(fp, "%d %d %d", &nx, &ny, &nz);
	printf("nx: %d, ny: %d, nz: %d\n", nx, ny, nz);
	float *grid = malloc(nx*ny*nz*sizeof(*grid));
	
	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			for (int k = 0; k < nz; k++) {
				fscanf(fp, "%f", &grid[k+ny*(j+nx*i)]);
				if (i < 3 && j < 3 && k < 3) {
					printf("Reading: %f ", grid[k+ny*(j+nx*i)]);
				}
			}
		}
	}
	fclose(fp);
	return grid;
}

int main() {
	float *grid = read_data("data/brightness_temp_6.dat");
	myps(40, grid, 100, "output/ps_6.dat");
	//float *grid = read_data("data/dummy.dat");
	//myps(3, grid, 100, "output/ps_dummy.dat");

	return 0;
}
