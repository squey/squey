/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include "PVSourceCreatorOpcUa.h"

#include "PVOpcUaSource.h"

PVRush::PVSourceCreatorOpcUa::source_p
PVRush::PVSourceCreatorOpcUa::create_source_from_input(PVInputDescription_p input) const
{
	source_p src(new PVRush::PVOpcUaSource(input));

	return src;
}

QString PVRush::PVSourceCreatorOpcUa::supported_type() const
{
	return QString("opcua");
}

bool PVRush::PVSourceCreatorOpcUa::pre_discovery(PVInputDescription_p /*input*/) const
{
	return true;
}

QString PVRush::PVSourceCreatorOpcUa::name() const
{
	return QString("opcua");
}
