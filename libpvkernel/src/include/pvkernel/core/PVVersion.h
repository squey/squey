/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVCORE_PVVERSION_FILE_H
#define PVCORE_PVVERSION_FILE_H

#include <QByteArray>
#include <pvbase/general.h>

namespace PVCore
{

class PVVersion
{
  public:
	static bool from_network_reply(QByteArray const& reply, version_t& current, version_t& last);
	static QString to_str(version_t v);
	static QString update_url();
};
}

#endif
