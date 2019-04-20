// -*- mode: c++; coding: utf-8-unix -*-
/// 
/// @file sim_conf.hh
///
///

#ifndef SIM_CONF_HH_
#define SIM_CONF_HH_

typedef float FLOAT_t;

const FLOAT_t T_low  = 20.0 + 273.15;
const FLOAT_t T_init = 20.0 + 273.15;
const FLOAT_t T_melt = 1723.0; /* [K] */
const FLOAT_t H_melt = 1929.0; /* [J/cm^3] */

const FLOAT_t D_SUS304 = 7.93; /* [g/cm^3] */
const FLOAT_t H_SUS304 = 0.59; /* [J/K/g] */
const FLOAT_t L_SUS304 = 0.167;
const FLOAT_t Cv_SUS304 = D_SUS304 * H_SUS304; /* [J/K/cm^3] */

const FLOAT_t DX = 1.0/256;
const FLOAT_t DY = 1.0/256;
const FLOAT_t DZ = 1.0/256;

const FLOAT_t DT = 1.0/16384;

const int NX = 256;
const int NY = 128;
const int NZ = 16;

const FLOAT_t RH_Y = 2000.0 * 0.01 * 0.01 * DY;

const FLOAT_t s_cx = L_SUS304 * DT / (DX * DX);
const FLOAT_t s_cy = L_SUS304 * DT / (DY * DY);
const FLOAT_t s_cz = L_SUS304 * DT / (DZ * DZ);
const FLOAT_t s_ry = RH_Y * DT / (DY * DY);

static inline int
crd2idx(int x, int y, int z)
{
    return (x * (NY*NZ) + y*NZ + z);
}

#endif /* SIM_CONF_HH_ */

// Local Variables:
// indent-tabs-mode: t
// tab-width: 4
// End:
