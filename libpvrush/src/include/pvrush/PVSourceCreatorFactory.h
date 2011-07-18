#ifndef PVRUSH_PVSOURCECREATORFACTORY_H
#define PVRUSH_PVSOURCECREATORFACTORY_H

#include <pvcore/general.h>
#include <pvrush/PVSourceCreator.h>
#include <pvrush/PVInputType.h>
#include <pvfilter/PVArgument.h>
#include <list>
#include <QHash>
#include <QString>

namespace PVRush {

typedef std::list<PVSourceCreator_p> list_creators;
typedef std::pair<PVFormat,PVSourceCreator_p> pair_format_creator;
typedef QHash<QString, pair_format_creator> hash_format_creator;

class LibRushDecl PVSourceCreatorFactory
{
public:
	static list_creators get_by_input_type(PVInputType_p in_t);
	static hash_format_creator get_supported_formats(list_creators const& lcr);
	static float discover_input(pair_format_creator format, PVFilter::PVArgument const& input);
};

}

#endif
