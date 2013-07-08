//abstracts the entire cuda functionality

#ifndef __DNSCUDAWRAPPER_H__
#define __DNSCUDAWRAPPER_H__

#include "param.h"

void cudaIterate(ptype *rho, ptype *u, ptype *v, ptype *w, ptype *p, ptype kt, ptype mu);

#endif

