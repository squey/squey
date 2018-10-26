/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2015-2015
 */

#include "PVSourceCreatorSplunk.h"
#include "PVSplunkSource.h"

PVRush::PVSourceCreatorSplunk::source_p
PVRush::PVSourceCreatorSplunk::create_source_from_input(PVInputDescription_p input) const
{
	source_p src(new PVRush::PVSplunkSource(input));

	return src;
}

QString PVRush::PVSourceCreatorSplunk::supported_type() const
{
	return QString("splunk");
}

bool PVRush::PVSourceCreatorSplunk::pre_discovery(PVInputDescription_p /*input*/) const
{
	return true;
}

QString PVRush::PVSourceCreatorSplunk::name() const
{
	return QString("splunk");
}
