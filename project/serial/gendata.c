#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#include "func.c"

#define MAX_NNB	256
#ifndef MANUAL
	#define PROBDIM 16
	#define NNBS 32
	#define TRAINELEMS 1048576
	#define QUERYELEMS 1024
	#define LOW  0
	#define HIGH 2
#endif

int main(int argc, char *argv[])
{
	double x[PROBDIM], y;
	FILE *fp;

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	SEED_RAND();	/* the training set is fixed */

	fp = fopen(trainfile, "w");

	for (int i=0;i<TRAINELEMS;i++)
	{
		for (int k = 0; k < PROBDIM; k++)
			x[k] = get_rand(k);

		y = fitfun(x, PROBDIM);

		for (int k = 0; k < PROBDIM; k++)
			fprintf(fp,"%.6f ", x[k]);

		fprintf(fp,"%.6f\n", y);
	}

	fclose(fp);
	printf("%d data points written to %s!\n", TRAINELEMS, trainfile);

	fp = fopen(queryfile, "w");
	for (int i=0;i<QUERYELEMS;i++)
	{
		for (int k = 0; k < PROBDIM; k++)
			x[k] = get_rand(k);

		y = fitfun(x, PROBDIM);

		for (int k = 0; k < PROBDIM; k++)
			fprintf(fp,"%.6f ", x[k]);

		fprintf(fp,"%.6f\n", y);
	}
	fclose(fp);
	printf("%d data points written to %s!\n", QUERYELEMS, queryfile);

	return 0;
}
