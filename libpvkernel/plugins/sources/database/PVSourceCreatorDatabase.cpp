/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVSourceCreatorDatabase.h"
#include "PVDBSource.h"
#include "../../common/database/PVDBQuery.h"

PVRush::PVSourceCreatorDatabase::source_p
PVRush::PVSourceCreatorDatabase::create_source_from_input(PVInputDescription_p input) const
{
	PVDBQuery* query = dynamic_cast<PVDBQuery*>(input.get());
	assert(query);
	source_p src = source_p(new PVRush::PVDBSource(*query, 100));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorDatabase::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorDatabase::supported_type() const
{
	return QString("database");
}

bool PVRush::PVSourceCreatorDatabase::pre_discovery(PVInputDescription_p /*input*/) const
{
	// There is only one database source for now
	return true;
}

QString PVRush::PVSourceCreatorDatabase::name() const
{
	return QString("database");
}
