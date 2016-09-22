/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVUNICODESOURCEERROR_H
#define PVRUSH_PVUNICODESOURCEERROR_H

#include <stdexcept>

namespace PVRush
{

/**
 * Exception throw from error with Unicode Source.
 */
class UnicodeSourceError : public std::runtime_error
{
	using std::runtime_error::runtime_error;
};
} // namespace PVRush

#endif
