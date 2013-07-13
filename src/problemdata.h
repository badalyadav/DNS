//Abstraction of problem data and CPU implementation of flux calculations
#ifndef __PROBLEMDATA_H__
#define __PROBLEMDATA_H__

#include<math.h>
#include "param.h"
#include "derivative.h"

#define ADDFLUX(FT, F, s, DF)	FOR(i, Np) FOR(j, 5) FT[j][i] = F[j][i] + s* DF[j][i];
#define ASSIGNFLUX(FT, F)	FOR(i, Np) FOR(j, 5) FT[j][i] = F[j][i];

//Problem is assumed in 3-dimensions
class ProblemData	
{
public:
	ptype dx;						//x(i+1) - x(i)
	ptype rho0, Mt, Re, k0, urms;	//mean-density, Mac number, Reynold's number, initial wave number with max energy, rms velocity
	ptype crms, lam, mu, p0, tau;	//rms speed of sound, length-scale, viscosity, mean-pressure, time-scale
	ptype kt;						//thermal conductivity
	
	ptype *W[5];									//conservative variables
	ptype *rho, *v, *u, *w, *p, *e, *Vsqr, *Csqr, *H, *T;	//primitive variables
		
	ProblemData(int N);
	~ProblemData();
	
	void p2c();
	void c2p();
	void c2p(ptype ** Wc);
	ptype getTimeStep();
	
	void derivs(ptype **DW, ptype ** Wc);	
		//uses the current staged value of conservative variables (RK scheme), 
		//converts it to primitive variable, 
		//and calculates the change of conservative variables (which is the sum of all fluxes)
		
	ptype *Wc[5];	//Conservative variables at a certain 'current' state of RK Scheme. 
	ptype *DW1[5], *DW2[5], *DW3[5], *DW4[5];
		
private:
	ptype *k_;	//a dummy array for constant '1' value
	ptype *ux, *uy, *uz, *vx, *vy, *vz, *wx, *wy, *wz;
	ptype *uxx, *uyy, *uzz, *uxy, *uyz, *uzx;
	ptype *vxx, *vyy, *vzz, *vxy, *vyz, *vzx;
	ptype *wxx, *wyy, *wzz, *wxy, *wyz, *wzx;
	ptype *px, *py, *pz;
	ptype *qxx, *qyy, *qzz;
	
	ptype *C[5];	//convective flux
	ptype *V[5];	//viscous flux
		
};


#endif
