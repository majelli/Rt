#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

double loglikelihood(int t, double R, int *data, int *imp, double **omega, int n);

int main(int argc, char *argv[]){
  char *seed = NULL;
  gsl_rng *gsl;
  int *data;
  int *imp;
  int n;
  double **omega;
  int i, j, e;
  double *shape;
  double *rate;
  double sigma;
  double **R;
  double Rtmp;
  double loglike, tmploglike;
  int nsim;
  char *tmp = NULL;
  	
  FILE *fp;

  if(argc != 6){
    fprintf(stderr,"Rt seed nsim sigma Tg data\n");
    exit(0);
  }
  
  seed = argv[1];
  nsim = atoi(argv[2]);
  sigma = atof(argv[3]);

  if(nsim <= 10000){
    fprintf(stderr,"nsim must be >10,000\n");
    exit(0);
  }

  //read time series
  fp = fopen(argv[5],"r");
  i = 0;
  tmp = (char *)malloc(100*sizeof(char));
  data = (int *)malloc(sizeof(int));
  imp = (int *)malloc(sizeof(int));  
  while( !feof(fp) ){
    fscanf(fp,"%s %d %d", tmp, &data[i], &imp[i]);
    i++;
    data = (int *)realloc(data, (i+1)*sizeof(int));
    imp = (int *)realloc(imp, (i+1)*sizeof(int));
  }
  n = i-1;

  // read Tg
  fp = fopen(argv[4],"r");
  i = 0;
  shape = (double *)malloc(sizeof(double));
  rate = (double *)malloc(sizeof(double));
  while( !feof(fp) ){
    fscanf(fp,"%lf %lf",&shape[i],&rate[i]);
    i++;
    shape = (double *)realloc(shape, (i+1)*sizeof(double));
    rate = (double *)realloc(rate, (i+1)*sizeof(double));
  }
  if( (i-1) < n ){
    fprintf(stderr,"Tg vector cannot be shorter than the length of the time series\n");
    exit(0);
  }
  
  //initialization of the seed
  setenv("GSL_RNG_SEED",seed, 1);
  gsl_rng_env_setup();
  gsl = gsl_rng_alloc (gsl_rng_default);

  //allocation of variables
  R = (double **)calloc(n,sizeof(double*)); 
  omega = (double**)calloc(n,sizeof(double*));		
  for(i=0; i<n; i++){
    R[i] = (double *)calloc(nsim,sizeof(double));
    omega[i] = (double *)calloc(n,sizeof(double));
  }

  //definition of the generation time
  for(i=0; i<n; i++){
    for(j=0; j<n; j++){
      omega[i][j] = gsl_ran_gamma_pdf(j, shape[i],1/rate[i]);
    }
  }    

  // loop over the time series
  for(i=1; i<n; i++){
    fprintf(stderr,"Day: %d\n",i);
    if( (data[i]-imp[i]) > 0){
      R[i][0] = 1.;
      loglike = loglikelihood(i,R[i][0],data,imp,omega,n);
    }else{
       R[i][0] = 0.;
    }
    
    // loop over the mcmc iterations 
    for(e=1; e<nsim; e++){
      R[i][e] = R[i][e-1];
      if( (data[i]-imp[i]) > 0){
	Rtmp = -1.;
	while(Rtmp < 0.){
	  Rtmp = R[i][e-1] + gsl_ran_gaussian(gsl,(1+data[i])*sigma); 
	}
	tmploglike = loglikelihood(i,Rtmp,data,imp,omega,n);
	if(gsl_ran_flat(gsl,0,1) < exp(tmploglike-loglike)){
	  loglike = tmploglike;
	  R[i][e] = Rtmp;
	}
      }
    }
  }
  
  for(e=nsim-10000; e<nsim; e++){
    fprintf(stdout,"%f",R[0][e]);
    for(i=1; i<n; i++){
      fprintf(stdout,"\t%f",R[i][e]);
    }
    fprintf(stdout,"\n");
  }
 
  return 0;
}
 	
double loglikelihood(int t, double R, int *data, int *imp, double **omega, int T){
  double loglike;
  double lambda;
  int s;
  double *omega2;
  double sumomega;

  omega2 = (double *)calloc(T,sizeof(double));
	
  sumomega = 0.;
  for(s=1; s<=t; s++){
    if(data[t-s] > 0){
      sumomega += omega[t-s][s]; 
    }
  }
  
  for(s=1; s<=t; s++){
    omega2[s] = omega[t-s][s]/sumomega;
  }
  
  lambda = 0.;
  for(s=1; s<=t; s++){
    lambda += data[t-s]*omega2[s];
  }
  
  loglike = log(gsl_ran_poisson_pdf(data[t]-imp[t],R*lambda)); 
	
  free(omega2);

  return loglike;	
}
