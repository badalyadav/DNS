#ifndef __DERIVATIVE_H__
#define __DERIVATIVE_H__

#include "param.h"

ptype Dx(ptype *A, int x, int y, int z);
ptype Dy(ptype *A, int x, int y, int z);
ptype Dz(ptype *A, int x, int y, int z);
ptype Dxx(ptype *A, int x, int y, int z);
ptype Dyy(ptype *A, int x, int y, int z);
ptype Dzz(ptype *A, int x, int y, int z);
ptype DABC(ptype *A, ptype *Bx, ptype *By, ptype *Bz, ptype*C, int x, int y, int z);

#endif
