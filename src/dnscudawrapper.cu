#include "dnscudawrapper.h"

//global macros
#define CUDAMAKE(var) cudaMalloc(&var, arrSize);	//maintains a double array for swapping
#define CUDAMAKEEXT(var) cudaMalloc(&var, arrSizeExt);	//maintains a double array for swapping
#define CUDAKILL(var) cudaFree(var);
#define CUDAMAKE5(var) cudaMalloc(&var, arrSize*5);
#define BLOCK_DIM 8

//Flag that determine the flow of program
#define PRINT_ENERGY

#define a1 0.79926643
#define a2 -0.18941314
#define a3 0.02651995
#define b1 1.5
#define b2 -0.6/4.0
#define b3 0.1/9.0

//cuda constant variables
__constant__ int devN, devNp;		//number of points
__constant__ int devNExt, devNpExt;	//number of points with margins extended to provide warping
__constant__ int devNTotal;			//Total number of points (irrespective of the gridN)
__constant__ ptype devkt;			//thermal conductivity
__constant__ ptype devmu;			//viscousity


//the data will be divided in cubic grids of size "gridN"
int gridN = 128;
int gridNp = gridN*gridN*gridN;
int gridNExt = gridN + BLOCK_DIM;						//extended grid including warped margins
int gridNpExt = gridNExt*gridNExt*gridNExt;	
	
//calculating standard array size in bytes
size_t arrSize  = gridNp * sizeof(ptype);
size_t arrSizeExt = gridNpExt * sizeof(ptype);

//Host function
void loadCubicData(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz);
void unloadCubicData(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz);
void loadCubicDataExt(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz);	//copying data to-&-from arrays with margins extended to provide warping
void unloadCubicDataExt(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz);
void loadCubicFlux(ptype *WHst[5], ptype *devW, int icx, int icy, int icz);
void unloadCubicFlux(ptype *WHst[5], ptype *devW, int icx, int icy, int icz);


//kernel functions
__global__ void kernel_p2qc(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W);	//p to q & c variables
__global__ void kernel_derives(ptype *rhoExt, ptype *uExt, ptype *vExt, ptype *wExt, ptype *pExt, ptype *HExt, ptype *TExt, ptype *W, 
								ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *DW, ptype f);
__global__ void kernel_derivesF(ptype *rhoExt, ptype *uExt, ptype *vExt, ptype *wExt, ptype *pExt, ptype *HExt, ptype *TExt, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, 
								ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W, ptype dt);
								
__global__ void kernel_derivesF(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, 
					ptype *W, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, ptype dt, bool swap);	//final step

//device functions
__device__ ptype D(ptype A_3, ptype A_2, ptype A_1, ptype A1, ptype A2, ptype A3);
__device__ ptype DD(ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3);
__device__ ptype DABC(  ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3,
						ptype B_3, ptype B_2, ptype B_1, ptype B0, ptype B1, ptype B2, ptype B3,
						ptype C_3, ptype C_2, ptype C_1, ptype C0, ptype C1, ptype C2, ptype C3 );


//central function for performing dns iteration on CUDA
void cudaIterate(ProblemData *prob)
{
	
	ptype *rhoTemp, *uTemp, *vTemp, *wTemp, *pTemp, *HTemp, *TTemp;
	ptype *WTemp[5];
	MAKE(rhoTemp);
	MAKE(uTemp);
	MAKE(vTemp);
	MAKE(wTemp);
	MAKE(pTemp);
	MAKE(HTemp);
	MAKE(TTemp);
	MAKE5(WTemp);
	
	ptype *devrho, *devu, *devv, *devw, *devp;							//primitive variables (referred as p variables)
	ptype *deve, *devH, *devT;											//extension of primitive variables... (referred as q variables)
	ptype *devVsqr, *devCsqr;											//square variables... (referred as q variables)
	ptype *devW;													//conservative variables	(referred as c variables)
	ptype *devDW;
	ptype *devDW1, *devDW2, *devDW3;									//change in flux
	ptype *devW0;
	
	ptype *devrhoExt, *devuExt, *devvExt, *devwExt, *devpExt, *devHExt, *devTExt;
	
	CUDAMAKE(devrho); CUDAMAKE(devu); CUDAMAKE(devv); CUDAMAKE(devw); CUDAMAKE(devp); 
	CUDAMAKE(deve); CUDAMAKE(devH); CUDAMAKE(devT);
	CUDAMAKE(devVsqr); CUDAMAKE(devCsqr);
	CUDAMAKE5(devW);
	CUDAMAKE5(devDW);
	CUDAMAKE5(devDW1); CUDAMAKE5(devDW2); CUDAMAKE5(devDW3); 
	CUDAMAKE5(devW0);
	
	CUDAMAKEEXT(devrhoExt); CUDAMAKEEXT(devuExt); CUDAMAKEEXT(devvExt); CUDAMAKEEXT(devwExt); CUDAMAKEEXT(devpExt);
	CUDAMAKEEXT(devHExt); CUDAMAKEEXT(devTExt); 
	
	//pointers used for loading & unloading memory chuncks
	ptype *primitiveHost[5] = {prob->rho, prob->u, prob->v, prob->w, prob->p};
	ptype *primitiveDev[5]  = {devrho, devu, devv, devw, devp};
	
	ptype *qHost[5] = {prob->e, prob->H, prob->T, prob->Vsqr, prob->Csqr};
	ptype *qDev[5]  = {deve, devH, devT, devVsqr, devCsqr};
	
	ptype *pqExtHost[7] = {rhoTemp, uTemp, vTemp, wTemp, pTemp, HTemp, TTemp};
	ptype *pqExtDev[7] = {devrhoExt, devuExt, devvExt, devwExt, devpExt, devHExt, devTExt};
	
	ptype *pqHost[8] = {prob->rho, prob->u, prob->v, prob->w, prob->p, prob->e, prob->H, prob->T};
	ptype *pqDev[8]  = {devrho, devu, devv, devw, devp, deve, devH, devT};
	
	ptype *pq2Host[10] = {prob->rho, prob->u, prob->v, prob->w, prob->p, prob->e, prob->H, prob->T, prob->Vsqr, prob->Csqr};
	ptype *pq2Dev[10]  = {devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr};
		
	//host variables used for calculating time-step
	ptype tc, tv, dt;
	ptype Et[TARGET_ITER], timeTotal[TARGET_ITER];
	ptype T=0;
	ptype Crms;
	ptype Vmax;
	ptype dx = 2*PI/N;
	
	//calculating thread and block count
	int gridDim = ceil((float)gridN/BLOCK_DIM); 
	dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM); 
	dim3 blocksPerGrid(gridDim, gridDim, gridDim);
	
	printf("\nCuda active... \n");
	printf("Block dim : %d	\nGrid dim : %d\n", BLOCK_DIM, gridDim);
	
	//calculation of cubic block counts
	int nC = ceil((float)N/gridN); 
	printf("No of cubic blocks : %dx%dx%d\n", nC, nC, nC);
	
	//loading constants
	cudaMemcpyToSymbol(devN , &gridN, sizeof(int));
	cudaMemcpyToSymbol(devNp, &gridNp, sizeof(int));
	cudaMemcpyToSymbol(devNExt , &gridNExt, sizeof(int));
	cudaMemcpyToSymbol(devNpExt, &gridNpExt, sizeof(int));
	cudaMemcpyToSymbol(devNTotal, &N, sizeof(int));
	cudaMemcpyToSymbol(devkt, &prob->kt, sizeof(ptype));
	cudaMemcpyToSymbol(devmu, &prob->mu, sizeof(ptype));
		
	//cuda events
	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	
	//calling kernel function for converting primitive variables to conservative
	FOR(icz, nC)
		FOR(icy, nC)
			FOR(icx, nC)
			{
				loadCubicData(primitiveHost, primitiveDev, 5, icx, icy, icz);
				kernel_p2qc<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr, devW);
				cudaThreadSynchronize();
				unloadCubicData(qHost, qDev, 5, icx, icy, icz);
				unloadCubicFlux(prob->W, devW, icx, icy, icz);
			}
	
	for(int iter=0;iter<TARGET_ITER;iter++)
	{
		
		Crms = 0;
		Vmax = 0;
		FOR(i, Np)
		{
			Crms += prob->Csqr[i];
			if(prob->Vsqr[i]>Vmax)
				Vmax = prob->Vsqr[i];
		}
		Crms = sqrt(Crms/Np);
		Vmax = sqrt(Vmax);
		tc = dx/(Vmax + Crms);
		tv = dx*dx/(2.0*prob->mu);
		dt = tv*tc/(tv+tc)*0.5;
		
		//step 1 of 4
		FOR(i, Np)
			{
				rhoTemp[i] = prob->rho[i];
				uTemp[i] = prob->u[i];
				vTemp[i] = prob->v[i];
				wTemp[i] = prob->w[i];
				pTemp[i] = prob->p[i];
				HTemp[i] = prob->H[i];
				TTemp[i] = prob->T[i];
			}
		FOR(icz, nC)
			FOR(icy, nC)
				FOR(icx, nC)
				{
					loadCubicDataExt(pqExtHost, pqExtDev, 7, icx, icy, icz);
					loadCubicFlux(prob->W, devW, icx, icy, icz);
					kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrhoExt, devuExt, devvExt, devwExt, devpExt, devHExt, devTExt, devW, 
								devrho, devu, devv, devw, devp, deve, devH, devT, devDW, 0.5*dt);
					unloadCubicFlux(prob->DW1, devDW, icx, icy, icz);
					unloadCubicData(pqHost, pqDev, 8, icx, icy, icz);
				}
		
		//step 2 of 4		
		FOR(i, Np)
			{
				rhoTemp[i] = prob->rho[i];
				uTemp[i] = prob->u[i];
				vTemp[i] = prob->v[i];
				wTemp[i] = prob->w[i];
				pTemp[i] = prob->p[i];
				HTemp[i] = prob->H[i];
				TTemp[i] = prob->T[i];
			}
		FOR(icz, nC)
			FOR(icy, nC)
				FOR(icx, nC)
				{
					loadCubicDataExt(pqExtHost, pqExtDev, 7, icx, icy, icz);
					loadCubicFlux(prob->W, devW, icx, icy, icz);
					kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrhoExt, devuExt, devvExt, devwExt, devpExt, devHExt, devTExt, devW, 
								devrho, devu, devv, devw, devp, deve, devH, devT, devDW, 0.5*dt);
					unloadCubicFlux(prob->DW2, devDW, icx, icy, icz);
					unloadCubicData(pqHost, pqDev, 8, icx, icy, icz);
				}
				
		//step 3 of 4		
		FOR(i, Np)
			{
				rhoTemp[i] = prob->rho[i];
				uTemp[i] = prob->u[i];
				vTemp[i] = prob->v[i];
				wTemp[i] = prob->w[i];
				pTemp[i] = prob->p[i];
				HTemp[i] = prob->H[i];
				TTemp[i] = prob->T[i];
			}
		FOR(icz, nC)
			FOR(icy, nC)
				FOR(icx, nC)
				{
					loadCubicDataExt(pqExtHost, pqExtDev, 7, icx, icy, icz);
					loadCubicFlux(prob->W, devW, icx, icy, icz);
					kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrhoExt, devuExt, devvExt, devwExt, devpExt, devHExt, devTExt, devW, 
								devrho, devu, devv, devw, devp, deve, devH, devT, devDW, 1.0*dt);
					unloadCubicFlux(prob->DW3, devDW, icx, icy, icz);
					unloadCubicData(pqHost, pqDev, 8, icx, icy, icz);
				}
				
		//step 4 of 4		
		FOR(i, Np)
			{
				rhoTemp[i] = prob->rho[i];
				uTemp[i] = prob->u[i];
				vTemp[i] = prob->v[i];
				wTemp[i] = prob->w[i];
				pTemp[i] = prob->p[i];
				HTemp[i] = prob->H[i];
				TTemp[i] = prob->T[i];
				WTemp[0][i] = prob->W[0][i];
				WTemp[1][i] = prob->W[1][i];
				WTemp[2][i] = prob->W[2][i];
				WTemp[3][i] = prob->W[3][i];
				WTemp[4][i] = prob->W[4][i];
			}
			
		FOR(icz, nC)
			FOR(icy, nC)
				FOR(icx, nC)
				{
					loadCubicDataExt(pqExtHost, pqExtDev, 7, icx, icy, icz);
					loadCubicFlux(WTemp, devW0, icx, icy, icz);
					loadCubicFlux(prob->DW1, devDW1, icx, icy, icz);
					loadCubicFlux(prob->DW2, devDW2, icx, icy, icz);
					loadCubicFlux(prob->DW3, devDW3, icx, icy, icz);
					kernel_derivesF<<<blocksPerGrid, threadsPerBlock>>>(devrhoExt, devuExt, devvExt, devwExt, devpExt, devHExt, devTExt, devW0, devDW1, devDW2, devDW3, 
								devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr, devW, dt);
					unloadCubicFlux(prob->W, devW, icx, icy, icz);
					unloadCubicData(pq2Host, pq2Dev, 10, icx, icy, icz);
				}
				
				
		#ifdef PRINT_ENERGY
			//calculation of total energy
			Et[iter] = 0;
			T += dt;
			FOR(i, Np)
			{
				Et[iter] += prob->Vsqr[i];
			}
			Et[iter] /= Np;
			timeTotal[iter] = T;
		#endif
	}
	
	printf("Last Cuda Error : %d\n", cudaGetLastError());

	print3DArray(prob->Vsqr);
	
	//capturing and timing events
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	
	cudaEventElapsedTime(&timeRecord.totalGPUTime, start, end);
	
	//freeing memory
	CUDAKILL(devrho); CUDAKILL(devu); CUDAKILL(devv); CUDAKILL(devw); CUDAKILL(devp); CUDAKILL(deve); CUDAKILL(devH); CUDAKILL(devT);
	CUDAKILL(devW); 
	CUDAKILL(devDW);
	CUDAKILL(devDW1); CUDAKILL(devDW2); CUDAKILL(devDW3);
	CUDAKILL(devW0);
	CUDAKILL(devrhoExt); CUDAKILL(devuExt); CUDAKILL(devvExt); CUDAKILL(devwExt); CUDAKILL(devpExt);
	CUDAKILL(devHExt); CUDAKILL(devTExt); 
	
	KILL(rhoTemp); KILL(uTemp); KILL(vTemp); KILL(wTemp); KILL(pTemp); KILL(HTemp); KILL(TTemp);
	KILL5(WTemp);
	
	//printing time-energy data to file
	#ifdef PRINT_ENERGY	
	string fileName = "Results/EnergyProfile_";
	fileName = fileName + TAG + ".csv";
	fstream energyFile(fileName.c_str(), ios::out | ios::trunc);
	energyFile<<"Time, Results\n";
	FOR(i, TARGET_ITER)
		energyFile<<timeTotal[i]<<", "<<Et[i]<<"\n";
	energyFile.close();
	#endif
	
}

void loadCubicData(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz)
{
	
	//creating temp array
	ptype *hstTemp[n];
	FOR(i, n)
		hstTemp[i] = new ptype[gridNp];
		
	FOR(iz, gridN)
		FOR(iy, gridN)
			FOR(ix, gridN)
			{
				int i1 = iz*gridN*gridN + iy*gridN + ix;
				
				int ix2 = (icx*gridN + ix)%N;
				int iy2 = (icy*gridN + iy)%N;
				int iz2 = (icz*gridN + iz)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, n)
					hstTemp[i][i1] = hst[i][i2];
				
			}	
	
	FOR(i, n)
	{
		cudaMemcpy(dev[i], hstTemp[i], arrSize, cudaMemcpyHostToDevice);	
		delete [] hstTemp[i];
	}
		
}

void unloadCubicData(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz)
{
	
	//creating temp array
	ptype *hstTemp[n];
	FOR(i, n)
	{
		hstTemp[i] = new ptype[gridNpExt];
		cudaMemcpy(hstTemp[i], dev[i], arrSize, cudaMemcpyDeviceToHost);	
	}
		
	FOR(iz, gridN)
		FOR(iy, gridN)
			FOR(ix, gridN)
			{
				int i1 = iz*gridN*gridN + iy*gridN + ix;
				
				int ix2 = (icx*gridN + ix)%N;
				int iy2 = (icy*gridN + iy)%N;
				int iz2 = (icz*gridN + iz)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, n)
					hst[i][i2] = hstTemp[i][i1];
			}	
	
	FOR(i, n)
		delete [] hstTemp[i];	
	
}

void loadCubicDataExt(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz)
{
	//creating temp array
	ptype *hstTemp[n];
	FOR(i, n)
		hstTemp[i] = new ptype[gridNpExt];
		
	FOR(iz, gridNExt)
		FOR(iy, gridNExt)
			FOR(ix, gridNExt)
			{
				int i1 = iz*gridNExt*gridNExt + iy*gridNExt + ix;
				
				int ix2 = (icx*gridN + ix - BLOCK_DIM/2 + N)%N;
				int iy2 = (icy*gridN + iy - BLOCK_DIM/2 + N)%N;
				int iz2 = (icz*gridN + iz - BLOCK_DIM/2 + N)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, n)
					hstTemp[i][i1] = hst[i][i2];
				
			}	
	
	FOR(i, n)
	{
		cudaMemcpy(dev[i], hstTemp[i], arrSizeExt, cudaMemcpyHostToDevice);	
		delete [] hstTemp[i];
	}
}

void unloadCubicDataExt(ptype *hst[], ptype *dev[], int n, int icx, int icy, int icz)
{
	//creating temp array
	ptype *hstTemp[n];
	FOR(i, n)
	{
		hstTemp[i] = new ptype[gridNpExt];
		cudaMemcpy(hstTemp[i], dev[i], arrSizeExt, cudaMemcpyDeviceToHost);	
	}
	
	FOR(iz, gridNExt)
		FOR(iy, gridNExt)
			FOR(ix, gridNExt)
			{
				int i1 = iz*gridNExt*gridNExt + iy*gridNExt + ix;
				
				int ix2 = (icx*gridN + ix - BLOCK_DIM/2 + N)%N;
				int iy2 = (icy*gridN + iy - BLOCK_DIM/2 + N)%N;
				int iz2 = (icz*gridN + iz - BLOCK_DIM/2 + N)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, n)
					hst[i][i2] = hstTemp[i][i1];
			}	
	
	FOR(i, n)
		delete [] hstTemp[i];
	
}

void loadCubicFlux(ptype *WHst[5], ptype *devW, int icx, int icy, int icz)
{
	
	ptype *WHstTemp = new ptype[gridNp*5];
	
	FOR(iz, gridN)
		FOR(iy, gridN)
			FOR(ix, gridN)
			{
				int i1 = iz*gridN*gridN + iy*gridN + ix;
				
				int ix2 = (icx*gridN + ix)%N;
				int iy2 = (icy*gridN + iy)%N;
				int iz2 = (icz*gridN + iz)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, 5)
					WHstTemp[i*gridNp + i1] = WHst[i][i2];
			}
	
	cudaMemcpy(devW, WHstTemp, arrSize*5, cudaMemcpyHostToDevice);	
	
}


void unloadCubicFlux(ptype *WHst[5], ptype *devW, int icx, int icy, int icz)
{
	ptype *WHstTemp = new ptype[gridNp*5];
	cudaMemcpy(WHstTemp, devW, arrSize*5, cudaMemcpyDeviceToHost);	
	
	FOR(iz, gridN)
		FOR(iy, gridN)
			FOR(ix, gridN)
			{
				int i1 = iz*gridN*gridN + iy*gridN + ix;
				
				int ix2 = (icx*gridN + ix)%N;
				int iy2 = (icy*gridN + iy)%N;
				int iz2 = (icz*gridN + iz)%N;
				int i2 = iz2*N*N + iy2*N + ix2;
				
				FOR(i, 5)
					WHst[i][i2] = WHstTemp[i*gridNp + i1];
			}
	
	
}


/////////// Kernel Functions
__global__ void kernel_p2qc(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W)
{

	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	int In = iZ*devN*devN + iY*devN + iX;
	
	Vsqr[In] = (u[In]*u[In] + v[In]*v[In] + w[In]*w[In]);
	Csqr[In] = GAMMA * GAMMA * p[In] * p[In] / (rho[In]*rho[In]);
	e[In] = p[In]/(rho[In]*(GAMMA-1)) + 0.5*Vsqr[In];
	H[In] = e[In] + p[In]/rho[In];
	T[In] = p[In]/(rho[In]*R);
	
	W[In] = rho[In];
	W[devNp*1 + In] = rho[In]*u[In];
	W[devNp*2 + In] = rho[In]*v[In];
	W[devNp*3 + In] = rho[In]*w[In];
	W[devNp*4 + In] = rho[In]*e[In];	
	
}


__global__ void kernel_derives(ptype *rhoExt, ptype *uExt, ptype *vExt, ptype *wExt, ptype *pExt, ptype *HExt, ptype *TExt, ptype *W, 
								ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *DW, ptype f)
{
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	int In = iZ*devN*devN + iY*devN + iX;
	
	///////// Calculation of all relevant indices
	//indices for accessing extended variable arrays
	int iX1 = (blockDim.x * blockIdx.x + threadIdx.x)%devNExt;
	int iY1 = (blockDim.y * blockIdx.y + threadIdx.y)%devNExt;
	int iZ1 = (blockDim.z * blockIdx.z + threadIdx.z)%devNExt;
	int iX_ = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM/2)%devNExt;
	int iY_ = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM/2)%devNExt;
	int iZ_ = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM/2)%devNExt;
	int iX2 = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM)%devNExt;
	int iY2 = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM)%devNExt;
	int iZ2 = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM)%devNExt;
	
	int In111 = iZ1*devNExt*devNExt + iY1*devNExt + iX1;
	int In112 = iZ1*devNExt*devNExt + iY1*devNExt + iX2;
	int In121 = iZ1*devNExt*devNExt + iY2*devNExt + iX1;
	int In122 = iZ1*devNExt*devNExt + iY2*devNExt + iX2;
	int In211 = iZ2*devNExt*devNExt + iY1*devNExt + iX1;
	int In212 = iZ2*devNExt*devNExt + iY1*devNExt + iX2;
	int In221 = iZ2*devNExt*devNExt + iY2*devNExt + iX1;
	int In222 = iZ2*devNExt*devNExt + iY2*devNExt + iX2;
	int In_   = iZ_*devNExt*devNExt + iY_*devNExt + iX_;
	
	int tx = threadIdx.x + BLOCK_DIM/2;	
	int ty = threadIdx.y + BLOCK_DIM/2;	
	int tz = threadIdx.z + BLOCK_DIM/2;	
	
	//declaring shared memory
	__shared__ ptype sh[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	float dx = 2*PI/devNTotal;
	
	__syncthreads();	
	
	//copying u into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = uExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = uExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = uExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = uExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = uExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = uExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = uExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = uExt[In222];
	
	__syncthreads();
	
	//derivatives of u
	ptype _u = sh[tz][ty][tx];
	ptype ux = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype uy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype uz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype uxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype uyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype uzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype uxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype uyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype uzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);
				
	__syncthreads();	
	
	//copying v into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = vExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = vExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = vExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = vExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = vExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = vExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = vExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = vExt[In222];
	
	__syncthreads();
	
	//derivatives of v
	ptype _v = sh[tz][ty][tx];
	ptype vx = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype vy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype vz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype vxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype vyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype vzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype vxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype vyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype vzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);
				
	__syncthreads();
	
	
	//copying w into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = wExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = wExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = wExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = wExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = wExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = wExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = wExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = wExt[In222];
	
	__syncthreads();
	
	//derivatives of w
	ptype _w = sh[tz][ty][tx];
	ptype wx = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype wy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype wz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype wxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype wyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype wzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype wxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype wyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype wzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);

	
	ptype px = D(	pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)],   
					pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)])/(dx);
	ptype py = D(	pExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)],
					pExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)])/(dx);
	ptype pz = D(	pExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					pExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)])/(dx);
					
	ptype qxx = -devkt * DD(	TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], TExt[In_],  
					TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)])/(dx*dx);
	ptype qyy = -devkt * DD(	TExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], TExt[In_], 
					TExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)])/(dx*dx);
	ptype qzz = -devkt * DD(	TExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[In_], 
					TExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)])/(dx*dx);
					
	
	//convective flux
	ptype C1 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				)/dx;
	
	ptype C2 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					uExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], 	uExt[In_], 
					uExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	uExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					uExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], uExt[In_], 
					uExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;
				
	ptype C3 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	vExt[In_],  
					vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					vExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], vExt[In_], 
					vExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;				
				
	ptype C4 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C5 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	HExt[In_],  
					HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					HExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], 	HExt[In_], 
					HExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	HExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					HExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], HExt[In_], 
					HExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;
					
					
	//viscous flux
	ptype V2 = devmu*( 2*uxx + uyy + uzz + vxy + wzx - 2.0/3.0*(uxx + vxy + wzx) );
	ptype V3 = devmu*( 2*vyy + vxx + vzz + uxy + wyz - 2.0/3.0*(uxy + vyy + wyz) );
	ptype V4 = devmu*( 2*wzz + wyy + wxx + vyz + uzx - 2.0/3.0*(uzx + vyz + wzz) );
	ptype V5 = 2*devmu*(  
			(uxx - (uxx+vxy+wzx)/3.0)*_u	+  	(ux - (ux+vy+wz)/3.0)*ux  
				+	(0.5*(uxy+vxx))*_v	+	(0.5*(uy+vx))*vx  
					+	(0.5*(uzx+wxx))*_w	+	(0.5*(uz+wx))*wx  
			+	(0.5*(uyy+vxy))*_u		+	(0.5*(uy+vx))*uy  
				+	(vyy - (uxy+vyy+wyz)/3.0)*_v	+	(vy - (ux+vy+wz)/3.0)*vy  
					+	(0.5*(wyy+vyz))*_w		+	(0.5*(wy+vz))*wy	 
			+	(0.5*(uzz+wzx))*_u		+	(0.5*(uz+wx))*uz  	
				+	(0.5*(wyz+vzz))*_v	+	(0.5*(wy+vz))*vz	 
					+	(wzz - (uzx+vyz+wzz)/3.0)*_w		+	(wz - (ux+vy+wz)/3.0)*wz  
			) - qxx - qyy - qzz;
	
	
	DW[In] 			 = - C1;
	DW[devNp*1 + In] = -(C2 + px - V2);
	DW[devNp*2 + In] = -(C3 + py - V3);
	DW[devNp*3 + In] = -(C4 + pz - V4);
	DW[devNp*4 + In] = -(C5 - V5);
	
	ptype w1 = W[In] - f*C1;
	ptype w2 = W[devNp*1 + In] - f*(C2 + px - V2);
	ptype w3 = W[devNp*2 + In] - f*(C3 + py - V3);
	ptype w4 = W[devNp*3 + In] - f*(C4 + pz - V4);
	ptype w5 = W[devNp*4 + In] - f*(C5 - V5);
	
	ptype rhoL = w1;
	ptype uL = w2/w1;
	ptype vL = w3/w1;
	ptype wL = w4/w1;
	ptype eL = w5/w1;
	ptype VsqrL = (uL*uL + vL*vL + wL*wL);
	ptype pL = (rhoL*(GAMMA-1))*(eL - 0.5*VsqrL);
	
	rho[In] = rhoL;
	u[In] = uL;
	v[In] = vL;
	w[In] = wL;
	e[In] = eL;
	p[In] = pL;
	H[In] = eL + pL/rhoL;
	T[In] = pL/(rhoL*R);
	
}


__global__ void kernel_derivesF(ptype *rhoExt, ptype *uExt, ptype *vExt, ptype *wExt, ptype *pExt, ptype *HExt, ptype *TExt, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, 
								ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W, ptype dt)
{
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	int In = iZ*devN*devN + iY*devN + iX;
	
	///////// Calculation of all relevant indices
	//indices for accessing extended variable arrays
	int iX1 = (blockDim.x * blockIdx.x + threadIdx.x)%devNExt;
	int iY1 = (blockDim.y * blockIdx.y + threadIdx.y)%devNExt;
	int iZ1 = (blockDim.z * blockIdx.z + threadIdx.z)%devNExt;
	int iX_ = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM/2)%devNExt;
	int iY_ = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM/2)%devNExt;
	int iZ_ = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM/2)%devNExt;
	int iX2 = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM)%devNExt;
	int iY2 = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM)%devNExt;
	int iZ2 = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM)%devNExt;
	
	int In111 = iZ1*devNExt*devNExt + iY1*devNExt + iX1;
	int In112 = iZ1*devNExt*devNExt + iY1*devNExt + iX2;
	int In121 = iZ1*devNExt*devNExt + iY2*devNExt + iX1;
	int In122 = iZ1*devNExt*devNExt + iY2*devNExt + iX2;
	int In211 = iZ2*devNExt*devNExt + iY1*devNExt + iX1;
	int In212 = iZ2*devNExt*devNExt + iY1*devNExt + iX2;
	int In221 = iZ2*devNExt*devNExt + iY2*devNExt + iX1;
	int In222 = iZ2*devNExt*devNExt + iY2*devNExt + iX2;
	int In_   = iZ_*devNExt*devNExt + iY_*devNExt + iX_;
	
	int tx = threadIdx.x + BLOCK_DIM/2;	
	int ty = threadIdx.y + BLOCK_DIM/2;	
	int tz = threadIdx.z + BLOCK_DIM/2;	
	
	//declaring shared memory
	__shared__ ptype sh[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	float dx = 2*PI/devNTotal;
	
	__syncthreads();	
	
	//copying u into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = uExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = uExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = uExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = uExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = uExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = uExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = uExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = uExt[In222];
	
	__syncthreads();
	
	//derivatives of u
	ptype _u = sh[tz][ty][tx];
	ptype ux = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype uy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype uz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype uxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype uyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype uzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype uxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype uyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype uzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);
				
	__syncthreads();	
	
	//copying v into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = vExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = vExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = vExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = vExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = vExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = vExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = vExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = vExt[In222];
	
	__syncthreads();
	
	//derivatives of v
	ptype _v = sh[tz][ty][tx];
	ptype vx = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype vy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype vz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype vxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype vyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype vzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype vxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype vyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype vzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);
				
	__syncthreads();
	
	
	//copying w into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = wExt[In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = wExt[In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = wExt[In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = wExt[In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = wExt[In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = wExt[In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = wExt[In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = wExt[In222];
	
	__syncthreads();
	
	//derivatives of w
	ptype _w = sh[tz][ty][tx];
	ptype wx = D(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/dx;
	ptype wy = D(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/dx;
	ptype wz = D(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/dx;
	ptype wxx = DD(sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3])/(dx*dx);
	ptype wyy = DD(sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx])/(dx*dx);
	ptype wzz = DD(sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx])/(dx*dx);
	ptype wxy = D(
					D(sh[tz][ty-3][tx-3], sh[tz][ty-3][tx-2], sh[tz][ty-3][tx-1], sh[tz][ty-3][tx+1], sh[tz][ty-3][tx+2], sh[tz][ty-3][tx+3]),
					D(sh[tz][ty-2][tx-3], sh[tz][ty-2][tx-2], sh[tz][ty-2][tx-1], sh[tz][ty-2][tx+1], sh[tz][ty-2][tx+2], sh[tz][ty-2][tx+3]),
					D(sh[tz][ty-1][tx-3], sh[tz][ty-1][tx-2], sh[tz][ty-1][tx-1], sh[tz][ty-1][tx+1], sh[tz][ty-1][tx+2], sh[tz][ty-1][tx+3]),
					D(sh[tz][ty+1][tx-3], sh[tz][ty+1][tx-2], sh[tz][ty+1][tx-1], sh[tz][ty+1][tx+1], sh[tz][ty+1][tx+2], sh[tz][ty+1][tx+3]),
					D(sh[tz][ty+2][tx-3], sh[tz][ty+2][tx-2], sh[tz][ty+2][tx-1], sh[tz][ty+2][tx+1], sh[tz][ty+2][tx+2], sh[tz][ty+2][tx+3]),
					D(sh[tz][ty+3][tx-3], sh[tz][ty+3][tx-2], sh[tz][ty+3][tx-1], sh[tz][ty+3][tx+1], sh[tz][ty+3][tx+2], sh[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype wyz = D(
					D(sh[tz-3][ty-3][tx], sh[tz-2][ty-3][tx], sh[tz-1][ty-3][tx], sh[tz+1][ty-3][tx], sh[tz+2][ty-3][tx], sh[tz+3][ty-3][tx]),
					D(sh[tz-3][ty-2][tx], sh[tz-2][ty-2][tx], sh[tz-1][ty-2][tx], sh[tz+1][ty-2][tx], sh[tz+2][ty-2][tx], sh[tz+3][ty-2][tx]),
					D(sh[tz-3][ty-1][tx], sh[tz-2][ty-1][tx], sh[tz-1][ty-1][tx], sh[tz+1][ty-1][tx], sh[tz+2][ty-1][tx], sh[tz+3][ty-1][tx]),
					D(sh[tz-3][ty+1][tx], sh[tz-2][ty+1][tx], sh[tz-1][ty+1][tx], sh[tz+1][ty+1][tx], sh[tz+2][ty+1][tx], sh[tz+3][ty+1][tx]),
					D(sh[tz-3][ty+2][tx], sh[tz-2][ty+2][tx], sh[tz-1][ty+2][tx], sh[tz+1][ty+2][tx], sh[tz+2][ty+2][tx], sh[tz+3][ty+2][tx]),
					D(sh[tz-3][ty+3][tx], sh[tz-2][ty+3][tx], sh[tz-1][ty+3][tx], sh[tz+1][ty+3][tx], sh[tz+2][ty+3][tx], sh[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype wzx = D(
					D(sh[tz-3][ty][tx-3], sh[tz-2][ty][tx-3], sh[tz-1][ty][tx-3], sh[tz+1][ty][tx-3], sh[tz+2][ty][tx-3], sh[tz+3][ty][tx-3]),
					D(sh[tz-3][ty][tx-2], sh[tz-2][ty][tx-2], sh[tz-1][ty][tx-2], sh[tz+1][ty][tx-2], sh[tz+2][ty][tx-2], sh[tz+3][ty][tx-2]),
					D(sh[tz-3][ty][tx-1], sh[tz-2][ty][tx-1], sh[tz-1][ty][tx-1], sh[tz+1][ty][tx-1], sh[tz+2][ty][tx-1], sh[tz+3][ty][tx-1]),
					D(sh[tz-3][ty][tx+1], sh[tz-2][ty][tx+1], sh[tz-1][ty][tx+1], sh[tz+1][ty][tx+1], sh[tz+2][ty][tx+1], sh[tz+3][ty][tx+1]),
					D(sh[tz-3][ty][tx+2], sh[tz-2][ty][tx+2], sh[tz-1][ty][tx+2], sh[tz+1][ty][tx+2], sh[tz+2][ty][tx+2], sh[tz+3][ty][tx+2]),
					D(sh[tz-3][ty][tx+3], sh[tz-2][ty][tx+3], sh[tz-1][ty][tx+3], sh[tz+1][ty][tx+3], sh[tz+2][ty][tx+3], sh[tz+3][ty][tx+3])
				)/(dx*dx);

	
	ptype px = D(	pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)],   
					pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], pExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)])/(dx);
	ptype py = D(	pExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)],
					pExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)], pExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)])/(dx);
	ptype pz = D(	pExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					pExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], pExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)])/(dx);
					
	ptype qxx = -devkt * DD(	TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], TExt[In_],  
					TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], TExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)])/(dx*dx);
	ptype qyy = -devkt * DD(	TExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], TExt[In_], 
					TExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)], TExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)])/(dx*dx);
	ptype qzz = -devkt * DD(	TExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[In_], 
					TExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], TExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)])/(dx*dx);
					
	
	//convective flux
	ptype C1 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				)/dx;
	
	ptype C2 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					uExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], 	uExt[In_], 
					uExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], 	uExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	uExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					uExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], uExt[In_], 
					uExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	uExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;
				
	ptype C3 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	vExt[In_],  
					vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	vExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					vExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], vExt[In_], 
					vExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	vExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;				
				
	ptype C4 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C5 = 	(
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	rhoExt[In_],  
					rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	rhoExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	uExt[In_],  
					uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	uExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)], 
					
					HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-3+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-2+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_-1+devNExt)%devNExt)], 	HExt[In_],  
					HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+1+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+2+devNExt)%devNExt)], 	HExt[(iZ_*devNExt*devNExt + iY_*devNExt + (iX_+3+devNExt)%devNExt)]
					)
				+
				DABC(
					rhoExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], rhoExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	rhoExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					vExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], vExt[In_], 
					vExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], vExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	vExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)],
					
					HExt[(iZ_*devNExt*devNExt + ((iY_-3+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_-2+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_-1+devNExt)%devNExt)*devNExt + iX_)], 	HExt[In_], 
					HExt[(iZ_*devNExt*devNExt + ((iY_+1+devNExt)%devNExt)*devNExt + iX_)], 	HExt[(iZ_*devNExt*devNExt + ((iY_+2+devNExt)%devNExt)*devNExt + iX_)],	HExt[(iZ_*devNExt*devNExt + ((iY_+3+devNExt)%devNExt)*devNExt + iX_)]
					)
				+
				DABC(
					rhoExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[In_], 
					rhoExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], rhoExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					HExt[(((iZ_-3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_-2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_-1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], HExt[In_], 
					HExt[(((iZ_+1+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_+2+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)], 	HExt[(((iZ_+3+devNExt)%devNExt)*devNExt*devNExt + iY_*devNExt + iX_)]
					)
				)/dx;
					
					
	//viscous flux
	ptype V2 = devmu*( 2*uxx + uyy + uzz + vxy + wzx - 2.0/3.0*(uxx + vxy + wzx) );
	ptype V3 = devmu*( 2*vyy + vxx + vzz + uxy + wyz - 2.0/3.0*(uxy + vyy + wyz) );
	ptype V4 = devmu*( 2*wzz + wyy + wxx + vyz + uzx - 2.0/3.0*(uzx + vyz + wzz) );
	ptype V5 = 2*devmu*(  
			(uxx - (uxx+vxy+wzx)/3.0)*_u	+  	(ux - (ux+vy+wz)/3.0)*ux  
				+	(0.5*(uxy+vxx))*_v	+	(0.5*(uy+vx))*vx  
					+	(0.5*(uzx+wxx))*_w	+	(0.5*(uz+wx))*wx  
			+	(0.5*(uyy+vxy))*_u		+	(0.5*(uy+vx))*uy  
				+	(vyy - (uxy+vyy+wyz)/3.0)*_v	+	(vy - (ux+vy+wz)/3.0)*vy  
					+	(0.5*(wyy+vyz))*_w		+	(0.5*(wy+vz))*wy	 
			+	(0.5*(uzz+wzx))*_u		+	(0.5*(uz+wx))*uz  	
				+	(0.5*(wyz+vzz))*_v	+	(0.5*(wy+vz))*vz	 
					+	(wzz - (uzx+vyz+wzz)/3.0)*_w		+	(wz - (ux+vy+wz)/3.0)*wz  
			) - qxx - qyy - qzz;
	
	
	ptype DW4_1	= - C1;
	ptype DW4_2	= -(C2 + px - V2);
	ptype DW4_3	= -(C3 + py - V3);
	ptype DW4_4	= -(C4 + pz - V4);
	ptype DW4_5	= -(C5 - V5);
	
	ptype w1	= W0[In] 		 + dt/6.0*(DW1[In]+2*DW2[In]+2*DW3[In]+DW4_1);
	ptype w2	= W0[devNp*1+In] + dt/6.0*(DW1[devNp*1+In]+2*DW2[devNp*1+In]+2*DW3[devNp*1+In]+DW4_2);
	ptype w3	= W0[devNp*2+In] + dt/6.0*(DW1[devNp*2+In]+2*DW2[devNp*2+In]+2*DW3[devNp*2+In]+DW4_3);
	ptype w4	= W0[devNp*3+In] + dt/6.0*(DW1[devNp*3+In]+2*DW2[devNp*3+In]+2*DW3[devNp*3+In]+DW4_4);
	ptype w5	= W0[devNp*4+In] + dt/6.0*(DW1[devNp*4+In]+2*DW2[devNp*4+In]+2*DW3[devNp*4+In]+DW4_5);
	
	ptype rhoL = w1;
	ptype uL = w2/w1;
	ptype vL = w3/w1;
	ptype wL = w4/w1;
	ptype eL = w5/w1;
	ptype VsqrL = (uL*uL + vL*vL + wL*wL);
	ptype pL = (rhoL*(GAMMA-1))*(eL - 0.5*VsqrL);
	
	rho[In] = rhoL;
	u[In] = uL;
	v[In] = vL;
	w[In] = wL;
	e[In] = eL;
	p[In] = pL;
	H[In] = eL + pL/rhoL;
	T[In] = pL/(rhoL*R);
	Vsqr[In] = VsqrL;
	Csqr[In] = GAMMA * GAMMA * pL * pL / (rhoL*rhoL);
	
	W[In] 			= w1;
	W[devNp*1+In] 	= w2;
	W[devNp*2+In] 	= w3;
	W[devNp*3+In] 	= w4;
	W[devNp*4+In] 	= w5;

}

//device functions
__device__ ptype D(ptype A_3, ptype A_2, ptype A_1, ptype A1, ptype A2, ptype A3)
{
	return (a1*(A1 - A_1) + a2*(A2 - A_2) + a3*(A3 - A_3));
}

__device__ ptype DD(ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3)
{
	return b1*(A1 + A_1) + b2*(A2 + A_2) + b3*(A3 + A_3) - 2*(b1+b2+b3)*A0;
	//return b1;
}

__device__ ptype DABC(
						ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3,
						ptype B_3, ptype B_2, ptype B_1, ptype B0, ptype B1, ptype B2, ptype B3,
						ptype C_3, ptype C_2, ptype C_1, ptype C0, ptype C1, ptype C2, ptype C3 )
{
	return 2.0*	(
				a1*0.125*((A0 + A1)*(B0 + B1)*(C0 + C1) - (A0 + A_1)*(B0 + B_1)*(C0 + C_1)) + 
				
				a2*0.125*((A0 + A2)*(B0 + B2)*(C0 + C2) - (A0 + A_2)*(B0 + B_2)*(C0 + C_2)) + 
				
				a3*0.125*((A0 + A3)*(B0 + B3)*(C0 + C3) - (A0 + A_3)*(B0 + B_3)*(C0 + C_3))
				);
}
