#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define MaxPher 0.8

typedef struct antcell{
	int row,col;
} Antcell;

// Print matrix of pherormones
void printPherormones(float * pherormones, int size)
{
	for(int i = 0; i< size; i++){
		for(int j=0; j< size; j++){
			printf("%.2f\t",pherormones[i*size+j]);
		}
	printf("\n");
	}
}

void printAnts(Antcell *antsPos, int antCount){
	for(int i = 0; i<antCount;i++)
	{
		printf("(%d,%d)\t",antsPos[i].row+1,antsPos[i].col+1);
	}
	printf("\n");
}

// Initialize pseudo-randomly the pherormones matrix and Ants positions
void initUnit(float *pherormones, int pher_dim, Antcell *antsPosOld, \
					Antcell *antsPosNew, int ant_dim, int *antCount)
{
	#pragma omp parallel
	{
		unsigned short randBuffer[3] = {0,0,time(0)+omp_get_thread_num()};
	
		// Pherormones Initialization
		#pragma omp for collapse(2)
		for(int i = 0; i< pher_dim; i++)
		{
			for(int j=0; j < pher_dim; j++)
			{	
				if( erand48(randBuffer) < 0.5 )
					pherormones[i*pher_dim+j] = MaxPher*erand48(randBuffer);
			}
		}
		
		// Ants positions Initializaton
		*antCount = 0;
		#pragma omp for collapse(2)
		for(int i=0; i<pher_dim; i++){
			for(int j=0; j<pher_dim; j+=5){
				if( erand48(randBuffer) <= 0.5){
					antsPosNew[*antCount].row = i;
					antsPosNew[*antCount].col = j+(int)4*erand48(randBuffer); 
					(*antCount)++;
				}
			}
		}
	
	}	//end of parallel region
}

void moveAnts(	float *pherormones, int pher_dim, Antcell *antsPosOld, \
				Antcell *antsPosNew, int ant_dim, int antCount )
{
    memcpy(antsPosOld,antsPosNew,ant_dim*sizeof(Antcell));
	
	int max_x, max_y;

	//loop over every ant
	for(int i = 0; i < antCount; i++){   

		// Defensive programming for matrix bounds
		int left 	= (antsPosNew[i].col==0)?0:antsPosNew[i].col-1;
		int right 	= (antsPosNew[i].col== (pher_dim-1))?(pher_dim-1):antsPosNew[i].col+1;
		int top 	= (antsPosNew[i].row==0)?0:antsPosNew[i].row-1;
		int bottom 	= (antsPosNew[i].row == (pher_dim-1))?(pher_dim-1):antsPosNew[i].row+1;

		// Find the next position 
		max_x = antsPosNew[i].row;
		max_y = left;

		if ( pherormones[max_x*pher_dim + max_y] < pherormones[antsPosNew[i].row*pher_dim + right] )
			max_y = right;
		if ( pherormones[max_x*pher_dim + max_y] < pherormones[top*pher_dim + antsPosNew[i].col] )
			max_x = top, max_y = antsPosNew[i].col;
		if ( pherormones[max_x*pher_dim + max_y] < pherormones[bottom*pher_dim + antsPosNew[i].col] )
			max_x = bottom, max_y = antsPosNew[i].col;

		// Check whether an ant already exists in that position
		char flag = 0;
		for(int j = 0; j < i; j++){
			if ( antsPosNew[j].row == max_x && antsPosNew[j].col == max_y){
				flag = 1;
				break;
			}
		}
		
		// Update ant with the new position
		antsPosNew[i].row = !(flag)? max_x: antsPosNew[i].row;
		antsPosNew[i].col = !(flag)? max_y: antsPosNew[i].col;
		pherormones[antsPosNew[i].row*pher_dim + antsPosNew[i].col] += MaxPher;
	}
}

void dispersePherormone(float *pherormones, int pher_dim, Antcell *antsPos, \
						int antCount, float dispersionRate ){
	//loop over every ant
	for(int i = 0; i < antCount; i++){
		// Defensive programming for matrix bounds
		int left 	= (antsPos[i].col==0)?0:antsPos[i].col-1;
		int right 	= (antsPos[i].col== (pher_dim-1))?(pher_dim-1):antsPos[i].col+1;
		int top 	= (antsPos[i].row==0)?0:antsPos[i].row-1;
		int bottom 	= (antsPos[i].row== (pher_dim-1))?(pher_dim-1):antsPos[i].row+1;

		// find the amount of pherormone to be dispersed
		float dispersedPherormone = pherormones[antsPos[i].row*pher_dim + antsPos[i].col] * dispersionRate;
		pherormones[antsPos[i].row*pher_dim + antsPos[i].col] -= dispersedPherormone;

		// Disperse the pherormone to adjacent cells.
		pherormones[antsPos[i].row*pher_dim + left] += dispersedPherormone * 0.25;
		pherormones[antsPos[i].row*pher_dim + right] += dispersedPherormone * 0.25;
		pherormones[top*pher_dim + antsPos[i].col] += dispersedPherormone * 0.25;
		pherormones[bottom*pher_dim + antsPos[i].col] += dispersedPherormone * 0.25;
	}
}

void evaporatePherormone(float *pherormones, int pher_dim, float evaporateRate, Antcell *antsPos, int antCount){
	char flag=0;
	for(int i = 0; i< pher_dim; i++){
		for(int j=0; j < pher_dim; j++){	
			for(int k=0; k < antCount; k++){
				
				if ((antsPos[k].row == i) && (antsPos[k].col == j)){
					flag = 1;
					break;
				}
				// If the code is run on cpu and if the condition is met
				// we should break the for loop. In the GPU the streaming
				// multiprocessor will execute every variable path of each
				// Thread. Will the rest of the threads executing wait cycles
				// We must avoid code branching if at all possible.
			}
			pherormones[i*pher_dim+j] *= (flag)? 1:evaporateRate;
		}
	}
}

int main(int argc, char ** argv){

	int antCount;
	int expCount = 200;
	
	int M = 50000;  // Ant size, needs to be at least N^2/5
	int N = 500;	// Pherormone matrix size
	
	Antcell *antsPosOld;	// Array of structs -> Old Indexed position of Ant in pherormone matrix
	Antcell *antsPosNew;	// Array of structs -> New Indexed position of Ant in pherormone matrix
	float *pherormones;

	antsPosOld = (Antcell*)malloc(M*sizeof(Antcell));
	antsPosNew = (Antcell*)malloc(M*sizeof(Antcell));
	pherormones= (float*)malloc(N*N*sizeof(float));

	omp_set_num_threads(4);
	initUnit(pherormones,N,antsPosOld,antsPosNew,M,&antCount);

	double t1, t0;
	t0 = omp_get_wtime();
	for ( int i=0; i < expCount; i++){
		moveAnts(pherormones,N,antsPosOld,antsPosNew,M,antCount);
		dispersePherormone(pherormones,N,antsPosOld,antCount,0.5);
		evaporatePherormone(pherormones,N,0.3,antsPosNew,antCount);
	}
	t1 = omp_get_wtime();

	printf("Matrix Dimension [%d] \t Ants Number [%d] \t Execution time [%lf]\n",N,antCount,t1 - t0 );	
	return 0;
}