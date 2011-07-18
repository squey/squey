#ifndef PVRUSH_PVPLUGINS_H
#define PVRUSH_PVPLUGINS_H

#include <pvcore/general.h>
#include <QString>

#define INPUT_TYPE_PREFIX "input_type"
#define SOURCE_PREFIX "source"

namespace PVRush {

class LibRushDecl PVPluginsLoad
{
public:
	static int load_all_plugins();
	static int load_input_type_plugins();
	static int load_source_plugins();
	static QString get_input_type_dir();
	static QString get_source_dir();
};

}

#endif
