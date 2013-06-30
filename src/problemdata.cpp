#include "problemdata.h"


//Problem data function definitions
//constructor
ProblemData::ProblemData(int N)
{
	::Np = N*N*N;
	
	//creating arrays
	
	//public variables
	MAKE5(W);
	MAKE(rho);
	MAKE(u);
	MAKE(v);
	MAKE(w);
	MAKE(p);
	MAKE(e);
	MAKE(Vsqr);
	MAKE(H);
	MAKE(T);
	
	//dummny constant array
	MAKE(k_);
	
	//private, and derivatives of variables
	MAKE(ux); MAKE(uy); MAKE(uz);
	MAKE(vx); MAKE(vy); MAKE(vz);
	MAKE(wx); MAKE(wy); MAKE(wz);
	
	MAKE(uxx); MAKE(uyy); MAKE(uzz);
	MAKE(uxy); MAKE(uyz); MAKE(uzx);	
	MAKE(vxx); MAKE(vyy); MAKE(vzz);
	MAKE(vxy); MAKE(vyz); MAKE(vzx);
	MAKE(wxx); MAKE(wyy); MAKE(wzz);
	MAKE(wxy); MAKE(wyz); MAKE(wzx);
	
	MAKE(px); MAKE(py); MAKE(pz);
	MAKE(qxx); MAKE(qyy); MAKE(qzz);
	
	//fluxes
	MAKE5(C); 
	MAKE5(V); 
	
	//RK scheme related ... 
	MAKE5(Wc);
	MAKE5(DW1); MAKE5(DW2); MAKE5(DW3); MAKE5(DW4);
	
	//initiating constant array
	FOR(i, Np)
		k_[i] = 1.0;
}


//destructor
ProblemData::~ProblemData()
{
	//killing arrays
	KILL5(W);
	KILL(rho);
	KILL(u);
	KILL(v);
	KILL(w);
	KILL(p);
	KILL(e);
	KILL(Vsqr);
	KILL(H);
	KILL(T);

	KILL(k_);
	
	KILL(ux); KILL(uy); KILL(uz);
	KILL(vx); KILL(vy); KILL(vz);
	KILL(wx); KILL(wy); KILL(wz);
	
	KILL(uxx); KILL(uyy); KILL(uzz);
	KILL(uxy); KILL(uyz); KILL(uzx);	
	KILL(vxx); KILL(vyy); KILL(vzz);
	KILL(vxy); KILL(vyz); KILL(vzx);
	KILL(wxx); KILL(wyy); KILL(wzz);
	KILL(wxy); KILL(wyz); KILL(wzx);
	
	KILL(px); KILL(py); KILL(pz);
	KILL(qxx); KILL(qyy); KILL(qzz);
	
	KILL5(C); 
	KILL5(V); 
	
	KILL5(Wc);
	KILL5(DW1); KILL5(DW2); KILL5(DW3); KILL5(DW4);
	
}


//primitive to conservative conversion
void ProblemData::p2c()
{
	//conversion loop
	FOR(i, Np)
		{
			W[0][i] = rho[i];
			W[1][i] = rho[i]*u[i];
			W[2][i] = rho[i]*v[i];
			W[3][i] = rho[i]*w[i];
			Vsqr[i] = (u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
			e[i] = p[i]/(rho[i]*(GAMMA-1)) + 0.5*Vsqr[i];
			W[4][i] = rho[i]*e[i];
		}
	
}


//conservative to primitive conversion
void ProblemData::c2p()
{
	//conversion loop
	FOR(i, Np)
		{
			rho[i] = W[0][i];
			u[i] = W[1][i]/rho[i];
			v[i] = W[2][i]/rho[i];
			w[i] = W[3][i]/rho[i];
			e[i] = W[4][i]/rho[i];
			Vsqr[i] = (u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
			p[i] = (e[i] - 0.5*Vsqr[i])*(rho[i]*(GAMMA-1));
			H[i] = e[i] + p[i]/rho[i];
			T[i] = p[i]/(rho[i]*R);
		}
}

//conservative to primitive conversion
void ProblemData::c2p(ptype ** Wc)
{
	//conversion loop
	FOR(i, Np)
		{
			rho[i] = Wc[0][i];
			u[i] = Wc[1][i]/rho[i];
			v[i] = Wc[2][i]/rho[i];
			w[i] = Wc[3][i]/rho[i];
			e[i] = Wc[4][i]/rho[i];
			Vsqr[i] = (u[i]*u[i] + v[i]*v[i] + w[i]*w[i]);
			p[i] = (e[i] - 0.5*Vsqr[i])*(rho[i]*(GAMMA-1));
			H[i] = e[i] + p[i]/rho[i];
			T[i] = p[i]/(rho[i]*R);
		}
		
	
}


//returns the time step of iteration
ptype ProblemData::getTimeStep()
{
	ptype Vmax = sqrt(max(Vsqr));
	
	//Calculating speed-of-sound's square
	ptype *Csqr = new ptype[Np];
	FOR(i, Np) Csqr[i] = GAMMA * GAMMA * p[i] * p[i] / (rho[i] * rho[i]);
	
	ptype Crms = sqrt(sum(Csqr)/Np);

	//time step
	ptype tc = dx/(Vmax + Crms);
	ptype tv = dx*dx/(2.0*mu);
	ptype dt = tv*tc/(tv+tc)*0.5;
			
	delete [] Csqr;
	
	return dt;
}


//calculates DW at a certain stage of R-K Scheme
void ProblemData::derivs(ptype ** DW, ptype ** Wc)
{
	
	//updating primitive variables
	c2p(Wc);
	
	//Finding derivatives
	FOR(z, N)
		FOR(y, N)
			FOR(x, N)
			{
				int i = I;
				
				//derivatives of velocity
				ux[i] = Dx(u, x, y, z)/dx; uy[i] = Dy(u, x, y, z)/dx; uz[i] = Dz(u, x, y, z)/dx;
				vx[i] = Dx(v, x, y, z)/dx; vy[i] = Dy(v, x, y, z)/dx; vz[i] = Dz(v, x, y, z)/dx;
				wx[i] = Dx(w, x, y, z)/dx; wy[i] = Dy(w, x, y, z)/dx; wz[i] = Dz(w, x, y, z)/dx;
				
				uxx[i] = Dxx(u, x, y, z)/(dx*dx); uyy[i] = Dyy(u, x, y, z)/(dx*dx); uzz[i] = Dzz(u, x, y, z)/(dx*dx); 
				vxx[i] = Dxx(v, x, y, z)/(dx*dx); vyy[i] = Dyy(v, x, y, z)/(dx*dx); vzz[i] = Dzz(v, x, y, z)/(dx*dx); 
				wxx[i] = Dxx(w, x, y, z)/(dx*dx); wyy[i] = Dyy(w, x, y, z)/(dx*dx); wzz[i] = Dzz(w, x, y, z)/(dx*dx); 
				
				//derivatives of pressure
				px[i] = Dx(p, x, y, z)/dx; py[i] = Dy(p, x, y, z)/dx; pz[i] = Dz(p, x, y, z)/dx;
				
				//heat flux
				qxx[i] = -kt*Dxx(T, x, y, z)/(dx*dx);
				qyy[i] = -kt*Dyy(T, x, y, z)/(dx*dx);
				qzz[i] = -kt*Dzz(T, x, y, z)/(dx*dx);
				
			}
			
	FOR(z, N)
		FOR(y, N)
			FOR(x, N)
			{
				int i = I;		
				
				//second derivatives
				uxy[i] = Dy(ux, x, y, z)/dx; uyz[i] = Dz(uy, x, y, z)/dx; uzx[i] = Dx(uz, x, y, z)/dx;
				vxy[i] = Dy(vx, x, y, z)/dx; vyz[i] = Dz(vy, x, y, z)/dx; vzx[i] = Dx(vz, x, y, z)/dx;
				wxy[i] = Dy(wx, x, y, z)/dx; wyz[i] = Dz(wy, x, y, z)/dx; wzx[i] = Dx(wz, x, y, z)/dx;
				
				//convective flux
				C[0][i] = DABC(rho, u, v, w, k_, x, y, z)/dx;
				C[1][i] = DABC(rho, u, v, w, u, x, y, z)/dx;
				C[2][i] = DABC(rho, u, v, w, v, x, y, z)/dx;
				C[3][i] = DABC(rho, u, v, w, w, x, y, z)/dx;
				C[4][i] = DABC(rho, u, v, w, H, x, y, z)/dx;
				
				//viscous flux
				V[0][i] = 0.0;
				V[1][i] = mu*( 2*uxx[i] + uyy[i] + uzz[i] + vxy[i] + wzx[i] - 2.0/3.0*(uxx[i] + vxy[i] + wzx[i]) );
				V[2][i] = mu*( 2*vyy[i] + vxx[i] + vzz[i] + uxy[i] + wyz[i] - 2.0/3.0*(uxy[i] + vyy[i] + wyz[i]) );
				V[3][i] = mu*( 2*wzz[i] + wyy[i] + wxx[i] + vyz[i] + uzx[i] - 2.0/3.0*(uzx[i] + vyz[i] + wzz[i]) );
				V[4][i] = 2*mu*(  
						(uxx[i] - (uxx[i]+vxy[i]+wzx[i])/3.0)*u[i]	+  	(ux[i] - (ux[i]+vy[i]+wz[i])/3.0)*ux[i]  
							+	(0.5*(uxy[i]+vxx[i]))*v[i]	+	(0.5*(uy[i]+vx[i]))*vx[i]  
								+	(0.5*(uzx[i]+wxx[i]))*w[i]	+	(0.5*(uz[i]+wx[i]))*wx[i]  
						+	(0.5*(uyy[i]+vxy[i]))*u[i]		+	(0.5*(uy[i]+vx[i]))*uy[i]  
							+	(vyy[i] - (uxy[i]+vyy[i]+wyz[i])/3.0)*v[i]	+	(vy[i] - (ux[i]+vy[i]+wz[i])/3.0)*vy[i]  
								+	(0.5*(wyy[i]+vyz[i]))*w[i]		+	(0.5*(wy[i]+vz[i]))*wy[i]	 
						+	(0.5*(uzz[i]+wzx[i]))*u[i]		+	(0.5*(uz[i]+wx[i]))*uz[i]  	
							+	(0.5*(wyz[i]+vzz[i]))*v[i]	+	(0.5*(wy[i]+vz[i]))*vz[i]	 
								+	(wzz[i] - (uzx[i]+vyz[i]+wzz[i])/3.0)*w[i]		+	(wz[i] - (ux[i]+vy[i]+wz[i])/3.0)*wz[i]  
						) - qxx[i] - qyy[i] - qzz[i];
						
				//total flux
				DW[0][i] = - C[0][i];
				DW[1][i] = -(C[1][i] + px[i] - V[1][i]);
				DW[2][i] = -(C[2][i] + py[i] - V[2][i]);
				DW[3][i] = -(C[3][i] + pz[i] - V[3][i]);
				DW[4][i] = -(C[4][i] - V[4][i]);
			}

}














