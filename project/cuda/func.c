#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* I/O routines */
FILE *open_traindata(char *trainfile)
{
	FILE *fp;

	fp = fopen(trainfile, "r");
	if (fp == NULL) {
		printf("traindata; File %s not available\n", trainfile);
		exit(1);
	}
	return fp;
}

FILE *open_querydata(char *queryfile)
{
	FILE *fp;

	fp = fopen(queryfile, "r");
	if (fp == NULL) {
		printf("querydata: File %s not available\n", queryfile);
		exit(1);
	}
	return fp;
}

double read_nextnum(FILE *fp)
{
	double val;

	int c = fscanf(fp, "%lf", &val);
	if (c <= 0) {
		fprintf(stderr, "fscanf returned %d\n", c);
		exit(1);
	}
			
	return val;
}

/* Timer */
double gettime()
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return (double) (tv.tv_sec+tv.tv_usec/1000000.0);
}

/* Function to approximate */
double fitfun(double *x, int n)
{
	double f = 0.0;
	int i;

#if 1
	for(i=0; i<n; i++)	/* circle */
		f += x[i]*x[i];
#endif
#if 0
	for(i=0; i<n-1; i++) {	/*  himmelblau */
		f = f + pow((x[i]*x[i]+x[i+1]-11.0),2) + pow((x[i]+x[i+1]*x[i+1]-7.0),2);
	}
#endif
#if 0
	for (i=0; i<n-1; i++)   /* rosenbrock */
		f = f + 100.0*pow((x[i+1]-x[i]*x[i]),2) + pow((x[i]-1.0),2);
#endif
#if 0
	for (i=0; i<n; i++)     /* rastrigin */
		f = f + pow(x[i],2) + 10.0 - 10.0*cos(2*M_PI*x[i]);
#endif

	return f;
}


/* random number generator  */
#define SEED_RAND()     srand48(1)
#define URAND()         drand48()

#ifndef LB
#define LB -1.0
#endif
#ifndef UB
#define UB 1.0
#endif

double get_rand(int k)
{
	return (UB-LB)*URAND()+LB;
}

float compute_mean(float *v, int n)
{
	int i;
	double s = 0;
	for (i = 0; i < n; i++) s += v[i];

	return s/n;
}

float compute_var(float *v, int n, float mean)
{
	int i;
	float s = 0;
	for (i = 0; i < n; i++) s += pow(v[i]-mean,2);

	return s/n;
}

float compute_max_pos(const float * __restrict__ v, const int n, int *pos)
{
	int i, p = 0;
	float vmax = v[0];
	for (i = 1; i < n; i++)
		if (v[i] > vmax) {
			vmax = v[i];
			p = i;
		}

	*pos = p;
	return vmax;
}
