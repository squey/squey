/**
 * \file bccb.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef OCL_BCCB_H
#define OCL_BCCB_H

#include <code_bz/types.h>
#include <code_bz/bcode_cb.h>

void ocl_bccb(const char* kernel_file, PVBCode* bcodes, size_t n, BCodeCB bccb);
void ocl_bccb2(const char* kernel_file, const PVBCode* bcodes, size_t n, BCodeCB bccb);

#endif
