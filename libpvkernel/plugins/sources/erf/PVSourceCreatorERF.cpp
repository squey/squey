/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include "PVSourceCreatorERF.h"
#include "PVERFSource.h"
#include "../../common/erf/PVERFDescription.h"

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorERF::source_p
PVRush::PVSourceCreatorERF::create_source_from_input(PVRush::PVInputDescription_p input) const
{
	source_p src(new PVRush::PVERFSource(input));

	return src;
}

QString PVRush::PVSourceCreatorERF::supported_type() const
{
	return QString("erf");
}

QString PVRush::PVSourceCreatorERF::name() const
{
	return QString("erf");
}
