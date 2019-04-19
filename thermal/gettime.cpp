/* -*- mode: c; coding: utf-8-unix -*- */
/**
 * @file gettime.cc
 * 
 */

#include <sys/time.h>
#include <stdlib.h>

#include "gettime.hh"

double
gettime( void )
{
    double s;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    s = ((double)tv.tv_usec) * 1e-6;
    s += (double)tv.tv_sec;

    return s;
}

/*
 * Local Variables:
 * indent-tabs-mode: t
 * tab-width: 4
 * End:
 */
