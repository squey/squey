/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <sigc++/sigc++.h>

namespace PVCore
{

struct PVDisconnector : public sigc::connection {
	using sigc::connection::connection;
	using sigc::connection::operator=;
	PVDisconnector(PVDisconnector&&) = delete;
	~PVDisconnector() { disconnect(); }
};

} // namespace PVCore