/**
 * \file PVSourceCreatorTexthdfs.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVSourceCreatorTexthdfs.h"
#include "../PVInputHDFS.h"
#include "../PVHadoopResultSource.h"
#include "../PVChunkAlignHadoop.h"
#include "../PVChunkTransformHadoop.h"

#include <pvkernel/filter/PVChunkFilter.h>
#include <pvkernel/rush/PVChunkTransform.h>

PVRush::PVSourceCreatorTexthdfs::source_p PVRush::PVSourceCreatorTexthdfs::create_source_from_input(input_type input, PVRush::PVFormat& used_format) const
{
	PVRush::PVInputHDFSFile *ihdfs = dynamic_cast<PVRush::PVInputHDFSFile*>(input.get());
	assert(ihdfs);
	/*if (!ihdfs.should_process_in_hadoop()) {
		return create_discovery_source_from_input(input);
	}*/

	PVLOG_DEBUG("(PVSourceCreatorTexthdfs::create_source_from_input) process source thanksto hadoop.\n");
	// Use hadoop to create the NRAW
	// The format needs to be changed to only include fields w/ no further processing, as Hadoop will do it !
	source_p src = source_p(new PVRush::PVHadoopResultSource(*ihdfs, used_format.get_axes().size(), 200000));

	// Only keep the axes in the format
	used_format.only_keep_axes();

	return src;
}

PVRush::PVSourceCreatorTexthdfs::source_p PVRush::PVSourceCreatorTexthdfs::create_discovery_source_from_input(input_type input, const PVFormat& /*format*/) const
{
	// input is a PVInputHDFSFile !
	PVInputHDFSFile* f = dynamic_cast<PVInputHDFSFile*>(input.get());
	assert(f);
	PVRush::PVInput_p ihdfs(new PVRush::PVInputHDFS(*f));
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

bool PVRush::PVSourceCreatorTexthdfs::pre_discovery(input_type /*input*/) const
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
