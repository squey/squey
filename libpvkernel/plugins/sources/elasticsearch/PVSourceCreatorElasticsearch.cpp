/**
 * \file PVSourceCreatorElasticsearch.cpp
 *
 * Copyright (C) Picviz Labs 2015
 */

#include "PVSourceCreatorElasticsearch.h"

#include <pvkernel/filter/PVChunkFilter.h>
#include "PVElasticsearchSource.h"

PVRush::PVSourceCreatorElasticsearch::source_p PVRush::PVSourceCreatorElasticsearch::create_discovery_source_from_input(PVInputDescription_p input, const PVFormat& /*format*/) const
{
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src(new PVRush::PVElasticsearchSource(input, chk_flt->f()));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorElasticsearch::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorElasticsearch::supported_type() const
{
	return QString("elasticsearch");
}

bool PVRush::PVSourceCreatorElasticsearch::pre_discovery(PVInputDescription_p /*input*/) const
{
	return true;
}

QString PVRush::PVSourceCreatorElasticsearch::name() const
{
	return QString("elasticsearch");
}
