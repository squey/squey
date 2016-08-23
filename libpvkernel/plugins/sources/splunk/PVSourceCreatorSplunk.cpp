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

PVRush::hash_formats PVRush::PVSourceCreatorSplunk::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
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
