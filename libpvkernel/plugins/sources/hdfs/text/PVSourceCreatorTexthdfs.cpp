#include "PVSourceCreatorTexthdfs.h"
#include "../PVInputHDFS.h"
#include "../PVInputHadoop.h"
#include "../PVChunkAlignHadoop.h"

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVChunkTransform.h>

PVRush::PVSourceCreatorTexthdfs::source_p PVRush::PVSourceCreatorTexthdfs::create_source_from_input(PVCore::PVArgument const& input) const
{
	PVRush::PVInputHDFSFile ihdfs = input.value<PVInputHDFSFile>();
	if (!ihdfs.should_process_in_hadoop()) {
		return create_discovery_source_from_input(input);
	}

	// Use hadoop to create the NRAW
	// The format needs to be changed to only include fields w/ no further processing, as Hadoop will do it !
	// Then, return a PVHadoopSource for that input.
	PVInputHadoop* ihadoop = new PVInputHadoop(ihdfs);
	PVInput_p phadoop(ihadoop);
	// TODO: hadoop chunk filter to create fields in //
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	PVChunkTransfrom* trans = new PVChunkTransform();
	source_p src = source_p(new PVRush::PVRawSource<>(phadoop, ihadoop->get_align(), 200000, trans, chk_flt->f()));
	return src;
}

PVRush::PVSourceCreatorTexthdfs::source_p PVRush::PVSourceCreatorTexthdfs::create_discovery_source_from_input(PVCore::PVArgument const& input) const
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
