#ifndef PVFILTER_PVPLUGINS_H
#define PVFILTER_PVPLUGINS_H

#include <pvcore/general.h>
#include <QString>

#define NORMALIZE_FILTER_PREFIX "normalize"

namespace PVFilter {

class LibFilterDecl PVPluginsLoad
{
public:
	static int load_all_plugins();
	static int load_normalize_plugins();
	static QString get_normalize_dir();
};

}

#endif
