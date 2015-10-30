/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVUTILS_H
#define PVCORE_PVUTILS_H

#include <pvkernel/core/general.h>

#ifdef __GCC__
	#define likely(x)   __builtin_expect((x),1)
	#define unlikely(x) __builtin_expect((x),0)
#else
	#define likely(x)   (x)
	#define unlikely(x) (x)
#endif

#endif	/* PVCORE_PVUTILS_H */
