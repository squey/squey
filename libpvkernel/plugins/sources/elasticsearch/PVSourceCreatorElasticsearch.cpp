/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVSourceCreatorElasticsearch.h"

#include "PVElasticsearchSource.h"

PVRush::PVSourceCreatorElasticsearch::source_p
PVRush::PVSourceCreatorElasticsearch::create_source_from_input(PVInputDescription_p input) const
{
	source_p src(new PVRush::PVElasticsearchSource(input));

	return src;
}

QString PVRush::PVSourceCreatorElasticsearch::supported_type() const
{
	return QString("elasticsearch");
}

QString PVRush::PVSourceCreatorElasticsearch::name() const
{
	return QString("elasticsearch");
}
