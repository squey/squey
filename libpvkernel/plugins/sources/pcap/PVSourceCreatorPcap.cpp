/**
 * @file
 *
 *
 * @copyright (C) ESI Group INENDI 2017
 */

#include "PVSourceCreatorPcap.h"
#include "PVPcapSource.h"
#include "../../common/pcap/PVPcapDescription.h"

#include <pvkernel/core/PVConfig.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVPcapsicum::PVSourceCreatorPcap::source_p
PVPcapsicum::PVSourceCreatorPcap::create_source_from_input(PVRush::PVInputDescription_p input) const
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	PVLOG_DEBUG("(pcap plugin) create source for %s\n", qPrintable(input->human_name()));
	PVRush::PVPcapDescription* pcap_desc = dynamic_cast<PVRush::PVPcapDescription*>(input.get());
	assert(pcap_desc);
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(pcap_desc->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	int size_chunk = pvconfig.value("pvkernel/max_size_chunk").toInt();
	if (size_chunk <= 0) {
		size_chunk = 4096 * 100; // Aligned on a page boundary (4ko)
	}
	source_p src(new PVRush::PVPcapSource(ifile, pcap_desc, size_chunk));

	return src;
}

QString PVPcapsicum::PVSourceCreatorPcap::supported_type() const
{
	return QString("pcap");
}

QString PVPcapsicum::PVSourceCreatorPcap::name() const
{
	return QString("pcap");
}
