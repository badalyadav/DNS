/*
 * DNS (Direct Numerical Simulation) implemented on CPU & GPU
 */


//headers
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "param.h"
#include "problemdata.h"

#ifdef USE_CUDA
	#include "dnscudawrapper.h"
#endif

using namespace std;


//global function prototypes
void initProblem();
void cpuIterate();


//global variables
ProblemData *problemData;


//main function
int main(int argc, char **argv)
{
	
	initProblem();
	timeRecord.N = N;
	
	cudaIterate(problemData->rho, problemData->u, problemData->v, problemData->w, problemData->p, problemData->kt, problemData->mu);
	//cpuIterate();
	
	return 0;
}


//initialize problem
void initProblem()
{
	
	//opening file with initial conditions
	ifstream initFile(INIT_FILE_NAME, ios::in | ios::binary);
	
	//grid size
	initFile.read((char*) &N, sizeof(int));
	problemData = new ProblemData(N);
	printf("Grid Size : %d\nTotal number of Points : %d\n", N, Np);
	
	
	//reading other constants
	double var;
	initFile.read((char*) &var, sizeof(double));
	problemData->rho0 = var;
	initFile.read((char*) &var, sizeof(double));
	problemData->Mt = var;
	initFile.read((char*) &var, sizeof(double));
	problemData->Re = var;
	initFile.read((char*) &var, sizeof(double));
	problemData->k0 = var;
	initFile.read((char*) &var, sizeof(double));
	problemData->urms = var;	
	printf("Mac Number : %f\nReynold's Number : %f\n", problemData->Mt, problemData->Re);
	
	
	//calculating other constants
	problemData->dx = 2.0*PI/N;
	problemData->crms = sqrt(3.0)*problemData->urms/problemData->Mt;
	problemData->p0 = (problemData->rho0 * problemData->crms * problemData->crms) / GAMMA;
	problemData->lam = 2.0/problemData->k0;
	problemData->mu = (problemData->urms * problemData->lam * problemData->rho0)/problemData->Re ;
	problemData->kt = problemData->mu * Cp/Pr;
	problemData->tau = problemData->lam/problemData->urms;
		
	
	//reading velocity field
	double *u = new double[Np];
	initFile.read((char*) u, sizeof(double)*Np);
	double *v = new double[Np];
	initFile.read((char*) v, sizeof(double)*Np);
	double *w = new double[Np];
	initFile.read((char*) w, sizeof(double)*Np);
	
	
	//assigning values to problemData variables (primitive)
	FOR(z, N)
		FOR(y, N)
			FOR(x, N)
			{
				int i = I;
				int i2 = z * N * N + x * N + y;
				problemData->rho[i] = problemData->rho0;
				problemData->u[i] = u[i2];	//files are written in this order by default in octave
				problemData->v[i] = v[i2];
				problemData->w[i] = w[i2];
				problemData->p[i] = problemData->p0;
			}
			
	//deleting temp velocity field
	delete [] u;
	delete [] v;
	delete [] w;
	
	//closing file
	initFile.close();
	
}


void cpuIterate()
{
	
		time_t start, end;
				
		start = clock();
	
		//CPU version
		ptype time = 0;
		int iter = 0;
		ptype Et;
		
		while(time<=TARGET_TIME && iter<TARGET_ITER)
		{
			
			//convert to conservative variables
			problemData->p2c();
			
			//calculating time step
			ptype dt = problemData->getTimeStep();
			
			//RK-4 Scheme: calculating intermediate fluxes
			ASSIGNFLUX(problemData->Wc, problemData->W);
			problemData->derivs(problemData->DW1, problemData->Wc);
			ADDFLUX(problemData->Wc, problemData->W, 0.5*dt, problemData->DW1);
			problemData->derivs(problemData->DW2, problemData->Wc);
			ADDFLUX(problemData->Wc, problemData->W, 0.5*dt, problemData->DW2);
			problemData->derivs(problemData->DW3, problemData->Wc);
			ADDFLUX(problemData->Wc, problemData->W, dt, problemData->DW3);
			problemData->derivs(problemData->DW4, problemData->Wc);
			
			//RK-4: time integration
			ADDFLUX(problemData->W, problemData->W, 1.0*dt/6.0, problemData->DW1);
			ADDFLUX(problemData->W, problemData->W, 2.0*dt/6.0, problemData->DW2);
			ADDFLUX(problemData->W, problemData->W, 2.0*dt/6.0, problemData->DW3);
			ADDFLUX(problemData->W, problemData->W, 1.0*dt/6.0, problemData->DW4);
			
			//convert from conservative variables to primitive
			problemData->c2p();
			
			//iteration increment
			iter++;
			time += dt;
			
			//calculation of total energy
			Et = sum(problemData->Vsqr)/Np;
			
			printf("Iteration : %d \t\t time : %f \t\t Energy : %f \n", iter, time, Et);
			
		}
		
		end = clock();
		
		timeRecord.totalCPUTime = (float)(end - start)/CLOCKS_PER_SEC * 1000.0;
		
}
