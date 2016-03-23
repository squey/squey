/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVArcsightSource.h"
#include "PVSourceCreatorArcsight.h"

PVRush::PVSourceCreatorArcsight::source_p PVRush::PVSourceCreatorArcsight::create_source_from_input(PVInputDescription_p input, const PVFormat& /*format*/) const
{
	source_p src(new PVRush::PVArcsightSource(input, 128));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorArcsight::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorArcsight::supported_type() const
{
	return QString("arcsight");
}

bool PVRush::PVSourceCreatorArcsight::pre_discovery(PVInputDescription_p /*input*/) const
{
	return true;
}

QString PVRush::PVSourceCreatorArcsight::name() const
{
	return QString("arcsight");
}
