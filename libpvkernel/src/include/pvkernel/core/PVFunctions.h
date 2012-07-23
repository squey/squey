/**
 * \file PVFunctions.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVFUNCTIONS_H
#define PVCORE_PVFUNCTIONS_H

namespace PVCore {

struct undefined_function
{
	inline operator bool() const { return false; }
	inline void operator()() const { } 
};

}

#endif
