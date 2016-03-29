/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVSourceCreatorPcapfile.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVInputPcap.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorPcapfile::source_p PVRush::PVSourceCreatorPcapfile::create_source_from_input(PVInputDescription_p input, const PVFormat& /*format*/) const
{
	// input is a QString !
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	PVRush::PVInput_p ipcap(new PVRush::PVInputPcap(file->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	source_p src = PVRush::PVRawSourceBase_p(new PVRush::PVRawSource<>(ipcap, 16000));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorPcapfile::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorPcapfile::supported_type() const
{
	return QString("file");
}

bool PVRush::PVSourceCreatorPcapfile::pre_discovery(PVInputDescription_p input) const
{
	pcap_t *pcaph;
	char errbuf[PCAP_ERRBUF_SIZE];

	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	pcaph = pcap_open_offline(file->path().toLocal8Bit().constData(), errbuf);
	if (!pcaph) {
		return false;
	}

	pcap_close(pcaph);

	return true;
}

QString PVRush::PVSourceCreatorPcapfile::name() const
{
	return QString("pcap");
}
