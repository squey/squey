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
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	PVLOG_DEBUG("(pcap plugin) create source for %s\n", qPrintable(input->human_name()));
	PVRush::PVERFDescription* erf_desc = dynamic_cast<PVRush::PVERFDescription*>(input.get());
	assert(erf_desc);
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(erf_desc->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	int size_chunk = pvconfig.value("pvkernel/max_size_chunk").toInt();
	if (size_chunk <= 0) {
		size_chunk = 4096 * 100; // Aligned on a page boundary (4ko)
	}
	source_p src(new PVRush::PVERFSource(ifile, erf_desc, size_chunk));

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
