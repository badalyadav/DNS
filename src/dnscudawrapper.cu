#include "dnscudawrapper.h"

//global macros
#define CUDAMAKE(var) cudaMalloc(&var, arrSize);
#define CUDAKILL(var) cudaFree(var);
#define CUDAMAKE5(var) cudaMalloc(&var, arrSize*5);
#define BLOCK_DIM 6

//Flag that determine the flow of program
#define PRINT_ENERGY

#define a1 0.79926643
#define a2 -0.18941314
#define a3 0.02651995
#define b1 1.5
#define b2 -0.6/4.0
#define b3 0.1/9.0

//cuda constant variables
__constant__ int devN, devNp;	//number of points
__constant__ ptype devkt;		//thermal conductivity
__constant__ ptype devmu;		//viscousity

//kernel functions
__global__ void kernel_p2qc(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W);	//p to q & c variables
__global__ void kernel_c2pq(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W);							//c to p & q variables
__global__ void kernel_c2pq(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W);	//c to p & q variables
__global__ void kernel_derives(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W, ptype *DW, ptype *Wc, ptype f);
__global__ void kernel_timeIntegrate(ptype *W, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, ptype *DW4, ptype dt);

//device functions
__device__ ptype D(ptype A_3, ptype A_2, ptype A_1, ptype A1, ptype A2, ptype A3);
__device__ ptype DD(ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3);
__device__ ptype DABC(  ptype A_3, ptype A_2, ptype A_1, ptype A0, ptype A1, ptype A2, ptype A3,
						ptype B_3, ptype B_2, ptype B_1, ptype B0, ptype B1, ptype B2, ptype B3,
						ptype C_3, ptype C_2, ptype C_1, ptype C0, ptype C1, ptype C2, ptype C3 );


//central function for performing dns iteration on CUDA
void cudaIterate(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype kt, ptype mu)
{
	//calculating standard array size in bytes
	size_t arrSize = Np * sizeof(ptype);
	
	//declaring device variables
	ptype *devrho, *devu, *devv, *devw, *devp;							//primitive variables (referred as p variables)
	ptype *deve, *devH, *devT; 											//extension of primitive variables... (referred as q variables)
	ptype *devVsqr, *devCsqr;											//square variables... (referred as q variables)
	ptype *devW;														//conservative variables	(referred as c variables)
	ptype *devDW1, *devDW2, *devDW3, *devDW4;							//change in flux
	ptype *devWc;
	
	CUDAMAKE(devrho); CUDAMAKE(devu); CUDAMAKE(devv); CUDAMAKE(devw); CUDAMAKE(devp); CUDAMAKE(deve); CUDAMAKE(devH); CUDAMAKE(devT);
	CUDAMAKE(devVsqr); CUDAMAKE(devCsqr);
	CUDAMAKE5(devW);
	CUDAMAKE5(devDW1); CUDAMAKE5(devDW2); CUDAMAKE5(devDW3); CUDAMAKE5(devDW4); CUDAMAKE5(devWc);
	
	//host variables
	ptype tc, tv, dt;
	ptype *Vsqr = new ptype[Np];
	ptype *Csqr = new ptype[Np];
	ptype Et[TARGET_ITER], timeTotal[TARGET_ITER];
	ptype T=0;
	ptype Crms;
	ptype Vmax;
	ptype dx = 2*PI/N;
	
	//calculating thread and block count
	int gridDim = ceil((float)N/BLOCK_DIM); 
	dim3 threadsPerBlock(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM); 
	dim3 blocksPerGrid(gridDim, gridDim, gridDim);
	
	printf("\nCuda active... \n");
	printf("Block dim : %d	\nGrid dim : %d\n", BLOCK_DIM, gridDim);
	
	//loading constants
	cudaMemcpyToSymbol(devN , &N, sizeof(int));
	cudaMemcpyToSymbol(devNp, &Np, sizeof(int));
	cudaMemcpyToSymbol(devkt, &kt, sizeof(ptype));
	cudaMemcpyToSymbol(devmu, &mu, sizeof(ptype));
		
	//cuda events
	cudaEvent_t start, startKernel1, endKernel1, end;
	cudaEventCreate(&start);
	cudaEventCreate(&startKernel1);
	cudaEventCreate(&endKernel1);
	cudaEventCreate(&end);
	
	cudaEventRecord(start);
	cudaEventSynchronize(start);
	
	//copying memory from host to device
	cudaMemcpy(devrho, rho, arrSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(devu, u, arrSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(devv, v, arrSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(devw, w, arrSize, cudaMemcpyHostToDevice);		
	cudaMemcpy(devp, p, arrSize, cudaMemcpyHostToDevice);		
	
	//mem copy finished
	cudaEventRecord(startKernel1);
	cudaEventSynchronize(startKernel1);
		
	//calling kernel function for converting primitive variables to conservative
	kernel_p2qc<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr, devW);
	cudaThreadSynchronize();
	
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
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW , devDW1, devWc, 0.5*dt);
		cudaThreadSynchronize();
		kernel_c2pq<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devWc);
		cudaThreadSynchronize();
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW, devDW2, devWc, 0.5*dt);
		cudaThreadSynchronize();
		kernel_c2pq<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devWc);
		cudaThreadSynchronize();
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW, devDW3, devWc, 1.0*dt);
		cudaThreadSynchronize();
		kernel_c2pq<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devWc);
		cudaThreadSynchronize();
		kernel_derives<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devW, devDW4, devWc, 0.0);
		cudaThreadSynchronize();
		
		//RK-4: time integration
		kernel_timeIntegrate<<<blocksPerGrid, threadsPerBlock>>>(devW, devWc, devDW1, devDW2, devDW3, devDW4, dt);
		cudaThreadSynchronize();
		
		kernel_c2pq<<<blocksPerGrid, threadsPerBlock>>>(devrho, devu, devv, devw, devp, deve, devH, devT, devVsqr, devCsqr, devW);
		cudaThreadSynchronize();
		
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
			/**/
		#endif
	}
	
	printf("Last Cuda Error : %d\n", cudaGetLastError());
	
	//kernel 1 finished
	cudaEventRecord(endKernel1);
	cudaEventSynchronize(endKernel1);
	
	//copying back primitve variables
	cudaMemcpy(rho, devrho, arrSize, cudaMemcpyDeviceToHost);		
	cudaMemcpy(u, devu, arrSize, cudaMemcpyDeviceToHost);		
	cudaMemcpy(v, devv, arrSize, cudaMemcpyDeviceToHost);		
	cudaMemcpy(w, devw, arrSize, cudaMemcpyDeviceToHost);		
	cudaMemcpy(p, devp, arrSize, cudaMemcpyDeviceToHost);
	
	
	//capturing and timing events
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	
	cudaEventElapsedTime(&timeRecord.memCopy1Time, start, startKernel1);
	cudaEventElapsedTime(&timeRecord.kernel1Time, startKernel1, endKernel1);
	cudaEventElapsedTime(&timeRecord.memCopy2Time, endKernel1, end);
	cudaEventElapsedTime(&timeRecord.totalGPUTime, start, end);
	
	//freeing memory
	CUDAKILL(devrho); CUDAKILL(devu); CUDAKILL(devv); CUDAKILL(devw); CUDAKILL(devp); CUDAKILL(deve); CUDAKILL(devH); CUDAKILL(devT);
	CUDAKILL(devW);
	CUDAKILL(devDW1); CUDAKILL(devDW2); CUDAKILL(devDW3); CUDAKILL(devDW4); CUDAKILL(devWc);
	
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

__global__ void kernel_c2pq(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W)
{
	
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	
	int In = iZ*devN*devN + iY*devN + iX;
	
	ptype w1 = W[In];
	ptype w2 = W[devNp*1 + In];
	ptype w3 = W[devNp*2 + In];
	ptype w4 = W[devNp*3 + In];
	ptype w5 = W[devNp*4 + In];
	
	ptype rhoL = w1;
	ptype uL = w2/w1;
	ptype vL = w3/w1;
	ptype wL = w4/w1;
	ptype eL = w5/w1;
	ptype Vsqr = (uL*uL + vL*vL + wL*wL);
	ptype pL = (rhoL*(GAMMA-1))*(eL - 0.5*Vsqr);
	
	rho[In] = rhoL;
	u[In] = uL;
	v[In] = vL;
	w[In] = wL;
	e[In] = eL;
	p[In] = pL;
	H[In] = eL + pL/rhoL;
	T[In] = pL/(rhoL*R);	
}


__global__ void kernel_c2pq(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *Vsqr, ptype *Csqr, ptype *W)
{
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	
	int In = iZ*devN*devN + iY*devN + iX;
		
	ptype w1 = W[In];
	ptype w2 = W[devNp*1 + In];
	ptype w3 = W[devNp*2 + In];
	ptype w4 = W[devNp*3 + In];
	ptype w5 = W[devNp*4 + In];
	
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
	
}


__global__ void kernel_derives(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype *e, ptype *H, ptype *T, ptype *W, ptype *DW, ptype *Wc, ptype f)
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
	__shared__ ptype sh1[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	__shared__ ptype sh2[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	__shared__ ptype sh3[BLOCK_DIM*2][BLOCK_DIM*2][BLOCK_DIM*2];
	float dx = 2*PI/devN;
	
	//copying u into shared memory	
	sh1[threadIdx.z][threadIdx.y][threadIdx.x] = u[In111];
	sh1[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[In112];
	sh1[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[In121];
	sh1[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[In122];
	sh1[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = u[In211];
	sh1[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = u[In212];
	sh1[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = u[In221];
	sh1[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = u[In222];
	
	//copying v into shared memory	
	sh2[threadIdx.z][threadIdx.y][threadIdx.x] = v[In111];
	sh2[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[In112];
	sh2[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[In121];
	sh2[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[In122];
	sh2[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = v[In211];
	sh2[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = v[In212];
	sh2[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = v[In221];
	sh2[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = v[In222];
	
	//copying w into shared memory	
	sh3[threadIdx.z][threadIdx.y][threadIdx.x] = w[In111];
	sh3[threadIdx.z][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[In112];
	sh3[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[In121];
	sh3[threadIdx.z][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[In122];
	sh3[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x] = w[In211];
	sh3[threadIdx.z+BLOCK_DIM][threadIdx.y][threadIdx.x+BLOCK_DIM] = w[In212];
	sh3[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x] = w[In221];
	sh3[threadIdx.z+BLOCK_DIM][threadIdx.y+BLOCK_DIM][threadIdx.x+BLOCK_DIM] = w[In222];
	
	__syncthreads();
	
	//derivatives of u
	ptype _u = sh1[tz][ty][tx];
	ptype ux = D(sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3])/dx;
	ptype uy = D(sh1[tz][ty-3][tx], sh1[tz][ty-2][tx], sh1[tz][ty-1][tx], sh1[tz][ty+1][tx], sh1[tz][ty+2][tx], sh1[tz][ty+3][tx])/dx;
	ptype uz = D(sh1[tz-3][ty][tx], sh1[tz-2][ty][tx], sh1[tz-1][ty][tx], sh1[tz+1][ty][tx], sh1[tz+2][ty][tx], sh1[tz+3][ty][tx])/dx;
	ptype uxx = DD(sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3])/(dx*dx);
	ptype uyy = DD(sh1[tz][ty-3][tx], sh1[tz][ty-2][tx], sh1[tz][ty-1][tx], sh1[tz][ty][tx], sh1[tz][ty+1][tx], sh1[tz][ty+2][tx], sh1[tz][ty+3][tx])/(dx*dx);
	ptype uzz = DD(sh1[tz-3][ty][tx], sh1[tz-2][ty][tx], sh1[tz-1][ty][tx], sh1[tz][ty][tx], sh1[tz+1][ty][tx], sh1[tz+2][ty][tx], sh1[tz+3][ty][tx])/(dx*dx);
	ptype uxy = D(
					D(sh1[tz][ty-3][tx-3], sh1[tz][ty-3][tx-2], sh1[tz][ty-3][tx-1], sh1[tz][ty-3][tx+1], sh1[tz][ty-3][tx+2], sh1[tz][ty-3][tx+3]),
					D(sh1[tz][ty-2][tx-3], sh1[tz][ty-2][tx-2], sh1[tz][ty-2][tx-1], sh1[tz][ty-2][tx+1], sh1[tz][ty-2][tx+2], sh1[tz][ty-2][tx+3]),
					D(sh1[tz][ty-1][tx-3], sh1[tz][ty-1][tx-2], sh1[tz][ty-1][tx-1], sh1[tz][ty-1][tx+1], sh1[tz][ty-1][tx+2], sh1[tz][ty-1][tx+3]),
					D(sh1[tz][ty+1][tx-3], sh1[tz][ty+1][tx-2], sh1[tz][ty+1][tx-1], sh1[tz][ty+1][tx+1], sh1[tz][ty+1][tx+2], sh1[tz][ty+1][tx+3]),
					D(sh1[tz][ty+2][tx-3], sh1[tz][ty+2][tx-2], sh1[tz][ty+2][tx-1], sh1[tz][ty+2][tx+1], sh1[tz][ty+2][tx+2], sh1[tz][ty+2][tx+3]),
					D(sh1[tz][ty+3][tx-3], sh1[tz][ty+3][tx-2], sh1[tz][ty+3][tx-1], sh1[tz][ty+3][tx+1], sh1[tz][ty+3][tx+2], sh1[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype uyz = D(
					D(sh1[tz-3][ty-3][tx], sh1[tz-2][ty-3][tx], sh1[tz-1][ty-3][tx], sh1[tz+1][ty-3][tx], sh1[tz+2][ty-3][tx], sh1[tz+3][ty-3][tx]),
					D(sh1[tz-3][ty-2][tx], sh1[tz-2][ty-2][tx], sh1[tz-1][ty-2][tx], sh1[tz+1][ty-2][tx], sh1[tz+2][ty-2][tx], sh1[tz+3][ty-2][tx]),
					D(sh1[tz-3][ty-1][tx], sh1[tz-2][ty-1][tx], sh1[tz-1][ty-1][tx], sh1[tz+1][ty-1][tx], sh1[tz+2][ty-1][tx], sh1[tz+3][ty-1][tx]),
					D(sh1[tz-3][ty+1][tx], sh1[tz-2][ty+1][tx], sh1[tz-1][ty+1][tx], sh1[tz+1][ty+1][tx], sh1[tz+2][ty+1][tx], sh1[tz+3][ty+1][tx]),
					D(sh1[tz-3][ty+2][tx], sh1[tz-2][ty+2][tx], sh1[tz-1][ty+2][tx], sh1[tz+1][ty+2][tx], sh1[tz+2][ty+2][tx], sh1[tz+3][ty+2][tx]),
					D(sh1[tz-3][ty+3][tx], sh1[tz-2][ty+3][tx], sh1[tz-1][ty+3][tx], sh1[tz+1][ty+3][tx], sh1[tz+2][ty+3][tx], sh1[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype uzx = D(
					D(sh1[tz-3][ty][tx-3], sh1[tz-2][ty][tx-3], sh1[tz-1][ty][tx-3], sh1[tz+1][ty][tx-3], sh1[tz+2][ty][tx-3], sh1[tz+3][ty][tx-3]),
					D(sh1[tz-3][ty][tx-2], sh1[tz-2][ty][tx-2], sh1[tz-1][ty][tx-2], sh1[tz+1][ty][tx-2], sh1[tz+2][ty][tx-2], sh1[tz+3][ty][tx-2]),
					D(sh1[tz-3][ty][tx-1], sh1[tz-2][ty][tx-1], sh1[tz-1][ty][tx-1], sh1[tz+1][ty][tx-1], sh1[tz+2][ty][tx-1], sh1[tz+3][ty][tx-1]),
					D(sh1[tz-3][ty][tx+1], sh1[tz-2][ty][tx+1], sh1[tz-1][ty][tx+1], sh1[tz+1][ty][tx+1], sh1[tz+2][ty][tx+1], sh1[tz+3][ty][tx+1]),
					D(sh1[tz-3][ty][tx+2], sh1[tz-2][ty][tx+2], sh1[tz-1][ty][tx+2], sh1[tz+1][ty][tx+2], sh1[tz+2][ty][tx+2], sh1[tz+3][ty][tx+2]),
					D(sh1[tz-3][ty][tx+3], sh1[tz-2][ty][tx+3], sh1[tz-1][ty][tx+3], sh1[tz+1][ty][tx+3], sh1[tz+2][ty][tx+3], sh1[tz+3][ty][tx+3])
				)/(dx*dx);
				
	
	//derivatives of v
	ptype _v = sh2[tz][ty][tx];
	ptype vx = D(sh2[tz][ty][tx-3], sh2[tz][ty][tx-2], sh2[tz][ty][tx-1], sh2[tz][ty][tx+1], sh2[tz][ty][tx+2], sh2[tz][ty][tx+3])/dx;
	ptype vy = D(sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx])/dx;
	ptype vz = D(sh2[tz-3][ty][tx], sh2[tz-2][ty][tx], sh2[tz-1][ty][tx], sh2[tz+1][ty][tx], sh2[tz+2][ty][tx], sh2[tz+3][ty][tx])/dx;
	ptype vxx = DD(sh2[tz][ty][tx-3], sh2[tz][ty][tx-2], sh2[tz][ty][tx-1], sh2[tz][ty][tx], sh2[tz][ty][tx+1], sh2[tz][ty][tx+2], sh2[tz][ty][tx+3])/(dx*dx);
	ptype vyy = DD(sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx])/(dx*dx);
	ptype vzz = DD(sh2[tz-3][ty][tx], sh2[tz-2][ty][tx], sh2[tz-1][ty][tx], sh2[tz][ty][tx], sh2[tz+1][ty][tx], sh2[tz+2][ty][tx], sh2[tz+3][ty][tx])/(dx*dx);
	ptype vxy = D(
					D(sh2[tz][ty-3][tx-3], sh2[tz][ty-3][tx-2], sh2[tz][ty-3][tx-1], sh2[tz][ty-3][tx+1], sh2[tz][ty-3][tx+2], sh2[tz][ty-3][tx+3]),
					D(sh2[tz][ty-2][tx-3], sh2[tz][ty-2][tx-2], sh2[tz][ty-2][tx-1], sh2[tz][ty-2][tx+1], sh2[tz][ty-2][tx+2], sh2[tz][ty-2][tx+3]),
					D(sh2[tz][ty-1][tx-3], sh2[tz][ty-1][tx-2], sh2[tz][ty-1][tx-1], sh2[tz][ty-1][tx+1], sh2[tz][ty-1][tx+2], sh2[tz][ty-1][tx+3]),
					D(sh2[tz][ty+1][tx-3], sh2[tz][ty+1][tx-2], sh2[tz][ty+1][tx-1], sh2[tz][ty+1][tx+1], sh2[tz][ty+1][tx+2], sh2[tz][ty+1][tx+3]),
					D(sh2[tz][ty+2][tx-3], sh2[tz][ty+2][tx-2], sh2[tz][ty+2][tx-1], sh2[tz][ty+2][tx+1], sh2[tz][ty+2][tx+2], sh2[tz][ty+2][tx+3]),
					D(sh2[tz][ty+3][tx-3], sh2[tz][ty+3][tx-2], sh2[tz][ty+3][tx-1], sh2[tz][ty+3][tx+1], sh2[tz][ty+3][tx+2], sh2[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype vyz = D(
					D(sh2[tz-3][ty-3][tx], sh2[tz-2][ty-3][tx], sh2[tz-1][ty-3][tx], sh2[tz+1][ty-3][tx], sh2[tz+2][ty-3][tx], sh2[tz+3][ty-3][tx]),
					D(sh2[tz-3][ty-2][tx], sh2[tz-2][ty-2][tx], sh2[tz-1][ty-2][tx], sh2[tz+1][ty-2][tx], sh2[tz+2][ty-2][tx], sh2[tz+3][ty-2][tx]),
					D(sh2[tz-3][ty-1][tx], sh2[tz-2][ty-1][tx], sh2[tz-1][ty-1][tx], sh2[tz+1][ty-1][tx], sh2[tz+2][ty-1][tx], sh2[tz+3][ty-1][tx]),
					D(sh2[tz-3][ty+1][tx], sh2[tz-2][ty+1][tx], sh2[tz-1][ty+1][tx], sh2[tz+1][ty+1][tx], sh2[tz+2][ty+1][tx], sh2[tz+3][ty+1][tx]),
					D(sh2[tz-3][ty+2][tx], sh2[tz-2][ty+2][tx], sh2[tz-1][ty+2][tx], sh2[tz+1][ty+2][tx], sh2[tz+2][ty+2][tx], sh2[tz+3][ty+2][tx]),
					D(sh2[tz-3][ty+3][tx], sh2[tz-2][ty+3][tx], sh2[tz-1][ty+3][tx], sh2[tz+1][ty+3][tx], sh2[tz+2][ty+3][tx], sh2[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype vzx = D(
					D(sh2[tz-3][ty][tx-3], sh2[tz-2][ty][tx-3], sh2[tz-1][ty][tx-3], sh2[tz+1][ty][tx-3], sh2[tz+2][ty][tx-3], sh2[tz+3][ty][tx-3]),
					D(sh2[tz-3][ty][tx-2], sh2[tz-2][ty][tx-2], sh2[tz-1][ty][tx-2], sh2[tz+1][ty][tx-2], sh2[tz+2][ty][tx-2], sh2[tz+3][ty][tx-2]),
					D(sh2[tz-3][ty][tx-1], sh2[tz-2][ty][tx-1], sh2[tz-1][ty][tx-1], sh2[tz+1][ty][tx-1], sh2[tz+2][ty][tx-1], sh2[tz+3][ty][tx-1]),
					D(sh2[tz-3][ty][tx+1], sh2[tz-2][ty][tx+1], sh2[tz-1][ty][tx+1], sh2[tz+1][ty][tx+1], sh2[tz+2][ty][tx+1], sh2[tz+3][ty][tx+1]),
					D(sh2[tz-3][ty][tx+2], sh2[tz-2][ty][tx+2], sh2[tz-1][ty][tx+2], sh2[tz+1][ty][tx+2], sh2[tz+2][ty][tx+2], sh2[tz+3][ty][tx+2]),
					D(sh2[tz-3][ty][tx+3], sh2[tz-2][ty][tx+3], sh2[tz-1][ty][tx+3], sh2[tz+1][ty][tx+3], sh2[tz+2][ty][tx+3], sh2[tz+3][ty][tx+3])
				)/(dx*dx);
				
	
	//derivatives of w
	ptype _w = sh3[tz][ty][tx];
	ptype wx = D(sh3[tz][ty][tx-3], sh3[tz][ty][tx-2], sh3[tz][ty][tx-1], sh3[tz][ty][tx+1], sh3[tz][ty][tx+2], sh3[tz][ty][tx+3])/dx;
	ptype wy = D(sh3[tz][ty-3][tx], sh3[tz][ty-2][tx], sh3[tz][ty-1][tx], sh3[tz][ty+1][tx], sh3[tz][ty+2][tx], sh3[tz][ty+3][tx])/dx;
	ptype wz = D(sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx])/dx;
	ptype wxx = DD(sh3[tz][ty][tx-3], sh3[tz][ty][tx-2], sh3[tz][ty][tx-1], sh3[tz][ty][tx], sh3[tz][ty][tx+1], sh3[tz][ty][tx+2], sh3[tz][ty][tx+3])/(dx*dx);
	ptype wyy = DD(sh3[tz][ty-3][tx], sh3[tz][ty-2][tx], sh3[tz][ty-1][tx], sh3[tz][ty][tx], sh3[tz][ty+1][tx], sh3[tz][ty+2][tx], sh3[tz][ty+3][tx])/(dx*dx);
	ptype wzz = DD(sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx])/(dx*dx);
	ptype wxy = D(
					D(sh3[tz][ty-3][tx-3], sh3[tz][ty-3][tx-2], sh3[tz][ty-3][tx-1], sh3[tz][ty-3][tx+1], sh3[tz][ty-3][tx+2], sh3[tz][ty-3][tx+3]),
					D(sh3[tz][ty-2][tx-3], sh3[tz][ty-2][tx-2], sh3[tz][ty-2][tx-1], sh3[tz][ty-2][tx+1], sh3[tz][ty-2][tx+2], sh3[tz][ty-2][tx+3]),
					D(sh3[tz][ty-1][tx-3], sh3[tz][ty-1][tx-2], sh3[tz][ty-1][tx-1], sh3[tz][ty-1][tx+1], sh3[tz][ty-1][tx+2], sh3[tz][ty-1][tx+3]),
					D(sh3[tz][ty+1][tx-3], sh3[tz][ty+1][tx-2], sh3[tz][ty+1][tx-1], sh3[tz][ty+1][tx+1], sh3[tz][ty+1][tx+2], sh3[tz][ty+1][tx+3]),
					D(sh3[tz][ty+2][tx-3], sh3[tz][ty+2][tx-2], sh3[tz][ty+2][tx-1], sh3[tz][ty+2][tx+1], sh3[tz][ty+2][tx+2], sh3[tz][ty+2][tx+3]),
					D(sh3[tz][ty+3][tx-3], sh3[tz][ty+3][tx-2], sh3[tz][ty+3][tx-1], sh3[tz][ty+3][tx+1], sh3[tz][ty+3][tx+2], sh3[tz][ty+3][tx+3])
				)/(dx*dx);
	ptype wyz = D(
					D(sh3[tz-3][ty-3][tx], sh3[tz-2][ty-3][tx], sh3[tz-1][ty-3][tx], sh3[tz+1][ty-3][tx], sh3[tz+2][ty-3][tx], sh3[tz+3][ty-3][tx]),
					D(sh3[tz-3][ty-2][tx], sh3[tz-2][ty-2][tx], sh3[tz-1][ty-2][tx], sh3[tz+1][ty-2][tx], sh3[tz+2][ty-2][tx], sh3[tz+3][ty-2][tx]),
					D(sh3[tz-3][ty-1][tx], sh3[tz-2][ty-1][tx], sh3[tz-1][ty-1][tx], sh3[tz+1][ty-1][tx], sh3[tz+2][ty-1][tx], sh3[tz+3][ty-1][tx]),
					D(sh3[tz-3][ty+1][tx], sh3[tz-2][ty+1][tx], sh3[tz-1][ty+1][tx], sh3[tz+1][ty+1][tx], sh3[tz+2][ty+1][tx], sh3[tz+3][ty+1][tx]),
					D(sh3[tz-3][ty+2][tx], sh3[tz-2][ty+2][tx], sh3[tz-1][ty+2][tx], sh3[tz+1][ty+2][tx], sh3[tz+2][ty+2][tx], sh3[tz+3][ty+2][tx]),
					D(sh3[tz-3][ty+3][tx], sh3[tz-2][ty+3][tx], sh3[tz-1][ty+3][tx], sh3[tz+1][ty+3][tx], sh3[tz+2][ty+3][tx], sh3[tz+3][ty+3][tx])
				)/(dx*dx);
	ptype wzx = D(
					D(sh3[tz-3][ty][tx-3], sh3[tz-2][ty][tx-3], sh3[tz-1][ty][tx-3], sh3[tz+1][ty][tx-3], sh3[tz+2][ty][tx-3], sh3[tz+3][ty][tx-3]),
					D(sh3[tz-3][ty][tx-2], sh3[tz-2][ty][tx-2], sh3[tz-1][ty][tx-2], sh3[tz+1][ty][tx-2], sh3[tz+2][ty][tx-2], sh3[tz+3][ty][tx-2]),
					D(sh3[tz-3][ty][tx-1], sh3[tz-2][ty][tx-1], sh3[tz-1][ty][tx-1], sh3[tz+1][ty][tx-1], sh3[tz+2][ty][tx-1], sh3[tz+3][ty][tx-1]),
					D(sh3[tz-3][ty][tx+1], sh3[tz-2][ty][tx+1], sh3[tz-1][ty][tx+1], sh3[tz+1][ty][tx+1], sh3[tz+2][ty][tx+1], sh3[tz+3][ty][tx+1]),
					D(sh3[tz-3][ty][tx+2], sh3[tz-2][ty][tx+2], sh3[tz-1][ty][tx+2], sh3[tz+1][ty][tx+2], sh3[tz+2][ty][tx+2], sh3[tz+3][ty][tx+2]),
					D(sh3[tz-3][ty][tx+3], sh3[tz-2][ty][tx+3], sh3[tz-1][ty][tx+3], sh3[tz+1][ty][tx+3], sh3[tz+2][ty][tx+3], sh3[tz+3][ty][tx+3])
				)/(dx*dx);
	
	ptype px = D(	p[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], p[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], p[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)],   
					p[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], p[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], p[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx);
	ptype py = D(	p[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], p[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], p[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)],
					p[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], p[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], p[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx);
	ptype pz = D(	p[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], p[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], p[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)],
					p[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], p[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], p[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx);
	
	ptype qxx = -devkt * DD(	T[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], T[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], T[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], T[In],  
					T[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], T[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], T[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)])/(dx*dx);
	ptype qyy = -devkt * DD(	T[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], T[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], T[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], T[In], 
					T[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], T[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)], T[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)])/(dx*dx);
	ptype qzz = -devkt * DD(	T[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], T[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], T[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], T[In], 
					T[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], T[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], T[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)])/(dx*dx);
					
	
	//convective flux
	ptype C1 = 	(
				DABC(
					rho[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[In],  
					rho[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[In], 
					rho[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				+
				DABC(
					rho[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[In], 
					rho[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx],
					
					1, 1, 1, 1, 1, 1, 1
					)
				)/dx;
	
	ptype C2 = 	(
				DABC(
					rho[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[In],  
					rho[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3],
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3]
					)
				+
				DABC(
					rho[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[In], 
					rho[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx],
					
					sh1[tz][ty-3][tx], sh1[tz][ty-2][tx], sh1[tz][ty-1][tx], sh1[tz][ty][tx], sh1[tz][ty+1][tx], sh1[tz][ty+2][tx], sh1[tz][ty+3][tx]
					)
				+
				DABC(
					rho[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[In], 
					rho[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx],
					
					sh1[tz-3][ty][tx], sh1[tz-2][ty][tx], sh1[tz-1][ty][tx], sh1[tz][ty][tx], sh1[tz+1][ty][tx], sh1[tz+2][ty][tx], sh1[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C3 = 	(
				DABC(
					rho[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[In],  
					rho[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3],
					
					sh2[tz][ty][tx-3], sh2[tz][ty][tx-2], sh2[tz][ty][tx-1], sh2[tz][ty][tx], sh2[tz][ty][tx+1], sh2[tz][ty][tx+2], sh2[tz][ty][tx+3]
					)
				+
				DABC(
					rho[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[In], 
					rho[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx]
					)
				+
				DABC(
					rho[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[In], 
					rho[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx],
					
					sh2[tz-3][ty][tx], sh2[tz-2][ty][tx], sh2[tz-1][ty][tx], sh2[tz][ty][tx], sh2[tz+1][ty][tx], sh2[tz+2][ty][tx], sh2[tz+3][ty][tx]
					)
				)/dx;				
				
	ptype C4 = 	(
				DABC(
					rho[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[In],  
					rho[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3],
					
					sh3[tz][ty][tx-3], sh3[tz][ty][tx-2], sh3[tz][ty][tx-1], sh3[tz][ty][tx], sh3[tz][ty][tx+1], sh3[tz][ty][tx+2], sh3[tz][ty][tx+3]
					)
				+
				DABC(
					rho[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[In], 
					rho[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx],
					
					sh3[tz][ty-3][tx], sh3[tz][ty-2][tx], sh3[tz][ty-1][tx], sh3[tz][ty][tx], sh3[tz][ty+1][tx], sh3[tz][ty+2][tx], sh3[tz][ty+3][tx]
					)
				+
				DABC(
					rho[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[In], 
					rho[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx]
					)
				)/dx;
				
	ptype C5 = 	(
				DABC(
					rho[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	rho[In],  
					rho[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	rho[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)], 
					
					sh1[tz][ty][tx-3], sh1[tz][ty][tx-2], sh1[tz][ty][tx-1], sh1[tz][ty][tx], sh1[tz][ty][tx+1], sh1[tz][ty][tx+2], sh1[tz][ty][tx+3],
					
					H[(iZ*devN*devN + iY*devN + (iX-3+devN)%devN)], 	H[(iZ*devN*devN + iY*devN + (iX-2+devN)%devN)], 	H[(iZ*devN*devN + iY*devN + (iX-1+devN)%devN)], 	H[In],  
					H[(iZ*devN*devN + iY*devN + (iX+1+devN)%devN)], 	H[(iZ*devN*devN + iY*devN + (iX+2+devN)%devN)], 	H[(iZ*devN*devN + iY*devN + (iX+3+devN)%devN)]
					)
				+
				DABC(
					rho[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], rho[In], 
					rho[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], rho[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	rho[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)],
					
					sh2[tz][ty-3][tx], sh2[tz][ty-2][tx], sh2[tz][ty-1][tx], sh2[tz][ty][tx], sh2[tz][ty+1][tx], sh2[tz][ty+2][tx], sh2[tz][ty+3][tx],
					
					H[(iZ*devN*devN + ((iY-3+devN)%devN)*devN + iX)], 	H[(iZ*devN*devN + ((iY-2+devN)%devN)*devN + iX)], 	H[(iZ*devN*devN + ((iY-1+devN)%devN)*devN + iX)], 	H[In], 
					H[(iZ*devN*devN + ((iY+1+devN)%devN)*devN + iX)], 	H[(iZ*devN*devN + ((iY+2+devN)%devN)*devN + iX)],	H[(iZ*devN*devN + ((iY+3+devN)%devN)*devN + iX)]
					)
				+
				DABC(
					rho[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], rho[In], 
					rho[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], rho[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)],
					
					sh3[tz-3][ty][tx], sh3[tz-2][ty][tx], sh3[tz-1][ty][tx], sh3[tz][ty][tx], sh3[tz+1][ty][tx], sh3[tz+2][ty][tx], sh3[tz+3][ty][tx],
					
					H[(((iZ-3+devN)%devN)*devN*devN + iY*devN + iX)], 	H[(((iZ-2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[(((iZ-1+devN)%devN)*devN*devN + iY*devN + iX)], H[In], 
					H[(((iZ+1+devN)%devN)*devN*devN + iY*devN + iX)], 	H[(((iZ+2+devN)%devN)*devN*devN + iY*devN + iX)], 	H[(((iZ+3+devN)%devN)*devN*devN + iY*devN + iX)]
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
	
	//rho[In] = ux + uy + uz + uxx + uyy + uzz + uxy + uyz + uzx + vx + vy + vz + vxx + vyy + vzz + vxy + vyz + vzx + wx + wy + wz + wxx + wyy + wzz + wxy + wyz + wzx;
	//		+ px + py + pz + qxx + qyy + qzz + V2 + V3 + V4 + V5 + C1 + C2 + C3 + C4 + C5;
	
	//rho[In] = C1;
	
	DW[In] 			 = - C1;
	DW[devNp*1 + In] = -(C2 + px - V2);
	DW[devNp*2 + In] = -(C3 + py - V3);
	DW[devNp*3 + In] = -(C4 + pz - V4);
	DW[devNp*4 + In] = -(C5 - V5);
	
	Wc[In] 			 = W[In] - f*C1;
	Wc[devNp*1 + In] = W[devNp*1 + In] - f*(C2 + px - V2);
	Wc[devNp*2 + In] = W[devNp*2 + In] - f*(C3 + py - V3);
	Wc[devNp*3 + In] = W[devNp*3 + In] - f*(C4 + pz - V4);
	Wc[devNp*4 + In] = W[devNp*4 + In] - f*(C5 - V5);
	
	//rho[In] = p[In];
		
}


__global__ void kernel_timeIntegrate(ptype *W, ptype *W0, ptype *DW1, ptype *DW2, ptype *DW3, ptype *DW4, ptype dt)
{
	
	//index
	int iX = (blockDim.x * blockIdx.x + threadIdx.x)%devN;
	int iY = (blockDim.y * blockIdx.y + threadIdx.y)%devN;
	int iZ = (blockDim.z * blockIdx.z + threadIdx.z)%devN;
	
	int In = iZ*devN*devN + iY*devN + iX;
	
	W[In] = W0[In] + dt/6.0*(DW1[In]+2*DW2[In]+2*DW3[In]+DW4[In]);
	W[devNp*1+In] = W0[devNp*1+In] + dt/6.0*(DW1[devNp*1+In]+2*DW2[devNp*1+In]+2*DW3[devNp*1+In]+DW4[devNp*1+In]);
	W[devNp*2+In] = W0[devNp*2+In] + dt/6.0*(DW1[devNp*2+In]+2*DW2[devNp*2+In]+2*DW3[devNp*2+In]+DW4[devNp*2+In]);
	W[devNp*3+In] = W0[devNp*3+In] + dt/6.0*(DW1[devNp*3+In]+2*DW2[devNp*3+In]+2*DW3[devNp*3+In]+DW4[devNp*3+In]);
	W[devNp*4+In] = W0[devNp*4+In] + dt/6.0*(DW1[devNp*4+In]+2*DW2[devNp*4+In]+2*DW3[devNp*4+In]+DW4[devNp*4+In]);
	
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
