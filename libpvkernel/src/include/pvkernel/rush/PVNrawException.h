/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVNRAWEXCEPTION_H
#define PVRUSH_PVNRAWEXCEPTION_H

#include <stdexcept>

namespace PVRush
{

struct PVNrawException : public std::runtime_error {
	using std::runtime_error::runtime_error;
};
}

#endif
