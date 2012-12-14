/**
 * \file PVSourceCreatorFactory.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVRUSH_PVSOURCECREATORFACTORY_H
#define PVRUSH_PVSOURCECREATORFACTORY_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVArgument.h>
#include <list>
#include <map>
#include <QHash>
#include <QString>

namespace PVRush {

typedef std::list<PVSourceCreator_p> list_creators;
typedef std::pair<PVFormat,PVSourceCreator_p> pair_format_creator;
typedef QHash<QString, pair_format_creator> hash_format_creator;

class LibKernelDecl PVSourceCreatorFactory
{
public:
	static list_creators get_by_input_type(PVInputType_p in_t);
	static hash_format_creator get_supported_formats(list_creators const& lcr);
	static float discover_input(pair_format_creator format, PVInputDescription_p input, bool *cancellation = nullptr);
	static std::multimap<float, pair_format_creator> discover_input(PVInputType_p input_type, PVInputDescription_p input);
	static list_creators filter_creators_pre_discovery(PVRush::list_creators const& lcr, PVInputDescription_p input);
};

}

#endif
