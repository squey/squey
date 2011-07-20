#include "PVSourceCreatorTexthdfs.h"
#include "../PVInputHDFS.h"

#include <pvfilter/PVChunkFilter.h>

PVRush::PVSourceCreatorTexthdfs::source_p PVRush::PVSourceCreatorTexthdfs::create_source_from_input(PVCore::PVArgument const& input) const
{
	// input is a PVInputHDFSFile !
	PVRush::PVInput_p ihdfs(new PVRush::PVInputHDFS(input.value<PVInputHDFSFile>()));
	// FIXME: chunk size must be computed somewhere once and for all !
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src = source_p(new PVRush::PVUnicodeSource<>(ihdfs, 16000, chk_flt->f()));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorTexthdfs::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorTexthdfs::supported_type() const
{
	return QString("hdfs");
}

bool PVRush::PVSourceCreatorTexthdfs::pre_discovery(PVCore::PVArgument const& /*input*/) const
{
	// AG: I don't know a magic method for being sure that a file is a text-file
	// We'll let the TBB filters work for the moment...
	
	// So, it always returns true.
	return true;
}

QString PVRush::PVSourceCreatorTexthdfs::name() const
{
	return QString("text");
}
