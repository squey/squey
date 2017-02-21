/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2017
 */

#include <pvkernel/core/PVVersion.h>

namespace PVCore
{

std::ostream& operator<<(std::ostream& out, const PVCore::PVVersion& v)
{
	return out << v.major() << "." << v.minor() << "." << v.revision();
}

} // namespace PVCore
