#include "PVSourceCreatorDatabase.h"
#include "PVDBSource.h"
#include "../../common/database/PVDBQuery.h"

#include <pvkernel/filter/PVChunkFilter.h>

PVRush::PVSourceCreatorDatabase::source_p PVRush::PVSourceCreatorDatabase::create_source_from_input(PVCore::PVArgument const& input) const
{
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src = source_p(new PVRush::PVDBSource(input.value<PVDBQuery>(), 10000, chk_flt->f()));

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

bool PVRush::PVSourceCreatorDatabase::pre_discovery(PVCore::PVArgument const& /*input*/) const
{
	// There is only one database source for now
	return true;
}

QString PVRush::PVSourceCreatorDatabase::name() const
{
	return QString("database");
}
