/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVRUSH_PVSOURCECREATORFACTORY_H
#define PVRUSH_PVSOURCECREATORFACTORY_H

#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/core/PVArgument.h>
#include <list>
#include <map>
#include <QHash>
#include <QString>

namespace PVRush
{

typedef std::list<PVSourceCreator_p> list_creators;
typedef std::pair<PVFormat, PVSourceCreator_p> pair_format_creator;
typedef QHash<QString, pair_format_creator> hash_format_creator;

class PVSourceCreatorFactory
{
  public:
	static PVSourceCreator_p get_by_input_type(PVInputType_p in_t);
	static float discover_input(pair_format_creator format,
	                            PVInputDescription_p input,
	                            bool* cancellation = nullptr);
};
} // namespace PVRush

#endif
