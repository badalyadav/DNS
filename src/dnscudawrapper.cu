#include "dnscudawrapper.h"

//global macros
#define CUDAMAKE(var) cudaMalloc(&var, arrSize);	//maintains a double array for swapping
#define CUDAMAKE_(var) cudaMalloc(&var, arrSize_);	//maintains a double array for swapping
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
__constant__ ptype devkt;			//thermal conductivity
__constant__ ptype devmu;			//viscousity


//the data will be divided in cubic grids of size "gridN"
int gridN = 8;
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

//kernel functions
__global__ void kernel_p2qc(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W);	//p to q & c variables
__global__ void kernel_derives(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W, ptype *DW, ptype f, bool swap);
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
	
	ptype *devrho, *devu, *devv, *devw, *devp;							//primitive variables (referred as p variables)
	ptype *deve, *devH, *devT;											//extension of primitive variables... (referred as q variables)
	ptype *devVsqr, *devCsqr;											//square variables... (referred as q variables)
	ptype *devW;														//conservative variables	(referred as c variables)
	ptype *devDW1, *devDW2, *devDW3;									//change in flux
	ptype *devW0;
	bool swap = false;													//flag used for swapping between the double arrays
	
	CUDAMAKE(devrho); CUDAMAKE(devu); CUDAMAKE(devv); CUDAMAKE(devw); CUDAMAKE(devp); 
	CUDAMAKE(deve); CUDAMAKE(devH); CUDAMAKE(devT);
	CUDAMAKE(devVsqr); CUDAMAKE(devCsqr);
	CUDAMAKE5(devW);
	CUDAMAKE5(devDW1); CUDAMAKE5(devDW2); CUDAMAKE5(devDW3); 
	CUDAMAKE5(devW0);
	
	//pointers used for loading & unloading memory chuncks
	ptype *primitiveHost[5] = {prob->rho, prob->u, prob->v, prob->w, prob->p};
	ptype *primitiveDev[5]  = {devrho, devu, devv, devw, devp};
	ptype *primitive2Host[5] = {prob->e, prob->H, prob->T, prob->Vsqr, prob->Csqr};
	ptype *primitive2Dev[5]  = {deve, devH, devT, devVsqr, devCsqr};
	
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
				unloadCubicData(primitive2Host, primitive2Dev, 5, icx, icy, icz);
			}
	
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
	printf("time step : %f\t", dt);
	
	/*
	for(int iter=0;iter<TARGET_ITER;iter++)
	{
		
		//calculation of time
		cudaMemcpy(Vsqr, devVsqr, arrSize, cudaMemcpyDeviceToHost);		
		cudaMemcpy(Csqr, devCsqr, arrSize, cudaMemcpyDeviceToHost);
		Crms = 0;
		Vmax = 0;
		FOR(i, Np)
		{
			Crms += Csqr[i];
			if(Vsqr[i]>Vmax)
				Vmax = Vsqr[i];
		}
		Crms = sqrt(Crms/Np);
		Vmax = sqrt(Vmax);
		tc = dx/(Vmax + Crms);
		tv = dx*dx/(2.0*mu);
		dt = tv*tc/(tv+tc)*0.5;
				
		
		//RK-4 Scheme: calculating intermediate fluxes
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW , devDW1, 0.5*dt, swap);
		cudaThreadSynchronize();
		swap = !swap;
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW, devDW2, 0.5*dt, swap);
		cudaThreadSynchronize();
		swap = !swap;
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW, devDW3, 1.0*dt, swap);
		cudaThreadSynchronize();
		swap = !swap;
		
		cudaMemcpy(devW0, devW, arrSize*5, cudaMemcpyDeviceToDevice);		
		
		kernel_derivesF<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr, devW, devW0, devDW1, devDW2, devDW3, dt, swap);
		cudaThreadSynchronize();
		swap = !swap;
		
		#ifdef PRINT_ENERGY
			//calculation of total energy
			cudaMemcpy(Vsqr, devVsqr, arrSize, cudaMemcpyDeviceToHost);
			cudaThreadSynchronize();
			Et[iter] = 0;
			T += dt;
			FOR(i, Np)
			{
				Et[iter] += Vsqr[i];
			}
			Et[iter] /= Np;
			timeTotal[iter] = T;
		#endif
	}
	*/
	
	printf("Last Cuda Error : %d\n", cudaGetLastError());
	
	
	//capturing and timing events
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	
	cudaEventElapsedTime(&timeRecord.totalGPUTime, start, end);
	
	//freeing memory
	CUDAKILL(devrho); CUDAKILL(devu); CUDAKILL(devv); CUDAKILL(devw); CUDAKILL(devp); CUDAKILL(deve); CUDAKILL(devH); CUDAKILL(devT);
	CUDAKILL(devW);
	CUDAKILL(devDW1); CUDAKILL(devDW2); CUDAKILL(devDW3);
	CUDAKILL(devW0);
	
	
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
		hstTemp[i] = new ptype[gridNp];
		
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
		hstTemp[i] = new ptype[gridNp];
		cudaMemcpy(dev[i], hstTemp[i], arrSizeExt, cudaMemcpyHostToDevice);	
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


__global__ void kernel_derives(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W, ptype *DW, ptype f, bool swap)
{

	///////// Calculation of all relevant indices
	int iX1 = (blockDim.x * blockIdx.x + threadIdx.x - BLOCK_DIM/2 + devN)%devN;
	int iY1 = (blockDim.y * blockIdx.y + threadIdx.y - BLOCK_DIM/2 + devN)%devN;
	int iZ1 = (blockDim.z * blockIdx.z + threadIdx.z - BLOCK_DIM/2 + devN)%devN;
	int iX2 = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM/2 + devN)%devN;
	int iY2 = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM/2 + devN)%devN;
	int iZ2 = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM/2 + devN)%devN;
	
	int In111 = iZ1*devN*devN + iY1*devN + iX1;
	int In112 = iZ1*devN*devN + iY1*devN + iX2;
	int In121 = iZ1*devN*devN + iY2*devN + iX1;
	int In122 = iZ1*devN*devN + iY2*devN + iX2;
	int In211 = iZ2*devN*devN + iY1*devN + iX1;
	int In212 = iZ2*devN*devN + iY1*devN + iX2;
	int In221 = iZ2*devN*devN + iY2*devN + iX1;
	int In222 = iZ2*devN*devN + iY2*devN + iX2;
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	int In = iZ*devN*devN + iY*devN + iX;
	
	int tx = threadIdx.x + BLOCK_DIM/2;	
	int ty = threadIdx.y + BLOCK_DIM/2;	
	int tz = threadIdx.z + BLOCK_DIM/2;	
	
	//declaring shared memory
	__shared__ ptype sh[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	float dx = 2*PI/devN;
	
	__syncthreads();	
	
	//copying u into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = u[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = u[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In222];
	
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
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = v[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = v[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In222];
	
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
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = w[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = w[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In222];
	
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
	
	ptype px = D(	p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)],   
					p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx);
	ptype py = D(	p[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)],
					p[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx);
	ptype pz = D(	p[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)],
					p[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx);
	
	ptype qxx = -devkt * DD(	T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], T[swap*devNp+In],  
					T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx*dx);
	ptype qyy = -devkt * DD(	T[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], T[swap*devNp+In], 
					T[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx*dx);
	ptype qzz = -devkt * DD(	T[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+In], 
					T[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx*dx);
					
	
	//convective flux
	ptype C1 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				)/dx;
	
	ptype C2 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					u[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], 	u[swap*devNp+In], 
					u[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	u[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					u[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], u[swap*devNp+In], 
					u[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
					)
				)/dx;
				
	ptype C3 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	v[swap*devNp+In],  
					v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					v[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
					)
				)/dx;				
				
	ptype C4 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C5 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	H[swap*devNp+In],  
					H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					H[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], 	H[swap*devNp+In], 
					H[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	H[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					H[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], H[swap*devNp+In], 
					H[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
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
	
	rho[(!swap)*devNp+In] = rhoL;
	u[(!swap)*devNp+In] = uL;
	v[(!swap)*devNp+In] = vL;
	w[(!swap)*devNp+In] = wL;
	e[(!swap)*devNp+In] = eL;
	p[(!swap)*devNp+In] = pL;
	H[(!swap)*devNp+In] = eL + pL/rhoL;
	T[(!swap)*devNp+In] = pL/(rhoL*R);
	//Vsqr[(!swap)*devNp+In] = VsqrL;
	//Csqr[(!swap)*devNp+In] = GAMMA * GAMMA * pL * pL / (rhoL*rhoL);

}


__global__ void kernel_derivesF(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, 
					ptype *W, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, ptype dt, bool swap)
{

	///////// Calculation of all relevant indices
	int iX1 = (blockDim.x * blockIdx.x + threadIdx.x - BLOCK_DIM/2 + devN)%devN;
	int iY1 = (blockDim.y * blockIdx.y + threadIdx.y - BLOCK_DIM/2 + devN)%devN;
	int iZ1 = (blockDim.z * blockIdx.z + threadIdx.z - BLOCK_DIM/2 + devN)%devN;
	int iX2 = (blockDim.x * blockIdx.x + threadIdx.x + BLOCK_DIM/2 + devN)%devN;
	int iY2 = (blockDim.y * blockIdx.y + threadIdx.y + BLOCK_DIM/2 + devN)%devN;
	int iZ2 = (blockDim.z * blockIdx.z + threadIdx.z + BLOCK_DIM/2 + devN)%devN;
	
	int In111 = iZ1*devN*devN + iY1*devN + iX1;
	int In112 = iZ1*devN*devN + iY1*devN + iX2;
	int In121 = iZ1*devN*devN + iY2*devN + iX1;
	int In122 = iZ1*devN*devN + iY2*devN + iX2;
	int In211 = iZ2*devN*devN + iY1*devN + iX1;
	int In212 = iZ2*devN*devN + iY1*devN + iX2;
	int In221 = iZ2*devN*devN + iY2*devN + iX1;
	int In222 = iZ2*devN*devN + iY2*devN + iX2;
	
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	int In = iZ*devN*devN + iY*devN + iX;
		
	int tx = threadIdx.x + BLOCK_DIM/2;	
	int ty = threadIdx.y + BLOCK_DIM/2;	
	int tz = threadIdx.z + BLOCK_DIM/2;	
	
	//declaring shared memory
	__shared__ ptype sh[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	float dx = 2*PI/devN;
	
	__syncthreads();	
	
	//copying u into shared memory	
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = u[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = u[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[swap*devNp+In222];
	
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
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = v[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = v[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[swap*devNp+In222];
	
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
	sh[threadIdx.z][threadIdx.y][threadIdx.x] = w[swap*devNp+In111];
	sh[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In112];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[swap*devNp+In121];
	sh[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In122];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = w[swap*devNp+In211];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In212];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[swap*devNp+In221];
	sh[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[swap*devNp+In222];
	
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
	
	ptype px = D(	p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)],   
					p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], p[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx);
	ptype py = D(	p[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)],
					p[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], p[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx);
	ptype pz = D(	p[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)],
					p[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], p[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx);
	
	ptype qxx = -devkt * DD(	T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], T[swap*devNp+In],  
					T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], T[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx*dx);
	ptype qyy = -devkt * DD(	T[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], T[swap*devNp+In], 
					T[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], T[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx*dx);
	ptype qzz = -devkt * DD(	T[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+In], 
					T[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], T[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx*dx);
					
	
	//convective flux
	ptype C1 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				)/dx;
	
	ptype C2 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					u[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], 	u[swap*devNp+In], 
					u[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], 	u[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	u[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					u[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], u[swap*devNp+In], 
					u[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	u[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
					)
				)/dx;
				
	ptype C3 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	v[swap*devNp+In],  
					v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	v[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					v[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	v[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
					)
				)/dx;				
				
	ptype C4 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh[tz][ty][tx-3], sh[tz][ty][tx-2], sh[tz][ty][tx-1], sh[tz][ty][tx], sh[tz][ty][tx+1], sh[tz][ty][tx+2], sh[tz][ty][tx+3]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh[tz][ty-3][tx], sh[tz][ty-2][tx], sh[tz][ty-1][tx], sh[tz][ty][tx], sh[tz][ty+1][tx], sh[tz][ty+2][tx], sh[tz][ty+3][tx]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C5 = 	(
				DABC(
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[swap*devNp+In],  
					rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	u[swap*devNp+In],  
					u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	u[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	H[swap*devNp+In],  
					H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	H[swap*devNp+(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					v[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], v[swap*devNp+In], 
					v[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], v[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	v[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					H[swap*devNp+(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], 	H[swap*devNp+In], 
					H[swap*devNp+(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], 	H[swap*devNp+(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	H[swap*devNp+(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+In], 
					rho[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh[tz-3][ty][tx], sh[tz-2][ty][tx], sh[tz-1][ty][tx], sh[tz][ty][tx], sh[tz+1][ty][tx], sh[tz+2][ty][tx], sh[tz+3][ty][tx],
					
					H[swap*devNp+(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], H[swap*devNp+In], 
					H[swap*devNp+(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[swap*devNp+(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
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
	
	ptype w1	= W0[In] 		+ dt/6.0*(DW1[In]+2*DW2[In]+2*DW3[In]+DW4_1);
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
	
	rho[(!swap)*devNp+In] = rhoL;
	u[(!swap)*devNp+In] = uL;
	v[(!swap)*devNp+In] = vL;
	w[(!swap)*devNp+In] = wL;
	e[(!swap)*devNp+In] = eL;
	p[(!swap)*devNp+In] = pL;
	H[(!swap)*devNp+In] = eL + pL/rhoL;
	T[(!swap)*devNp+In] = pL/(rhoL*R);
	Vsqr[(!swap)*devNp+In] = VsqrL;
	Csqr[(!swap)*devNp+In] = GAMMA * GAMMA * pL * pL / (rhoL*rhoL);
	
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
