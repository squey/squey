/**
 * \file PVVersion.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVCORE_PVVERSION_FILE_H
#define PVCORE_PVVERSION_FILE_H

#include <pvkernel/core/general.h>
#include <QByteArray>

namespace PVCore {

class LibKernelDecl PVVersion
{
public:
	static bool from_network_reply(QByteArray const& reply, version_t& current, version_t& last);
	static QString to_str(version_t v);
	static QString update_url();
};

}

#endif

