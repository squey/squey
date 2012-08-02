/**
 * \file PVPluginsLoad.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVFILTER_PVPLUGINS_H
#define PVFILTER_PVPLUGINS_H

#include <pvkernel/core/general.h>
#include <QString>

#define NORMALIZE_FILTER_PREFIX "normalize"

namespace PVFilter {

class LibKernelDecl PVPluginsLoad
{
public:
	static int load_all_plugins();
	static int load_normalize_plugins();
	static QString get_normalize_dir();
};

}

#endif
