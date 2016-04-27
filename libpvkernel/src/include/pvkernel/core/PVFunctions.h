/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVFUNCTIONS_H
#define PVCORE_PVFUNCTIONS_H

namespace PVCore
{

struct undefined_function
{
	inline operator bool() const { return false; }
	inline void operator()() const {}
};
}

#endif
