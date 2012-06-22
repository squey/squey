#include "PVSourceCreatorPcapfile.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVInputPcap.h>
#include <pvkernel/rush/PVNormalizer.h>

#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorPcapfile::source_p PVRush::PVSourceCreatorPcapfile::create_discovery_source_from_input(PVInputDescription_p input, const PVFormat& /*format*/) const
{
	// input is a QString !
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	PVRush::PVChunkTransform* transform_null = new PVRush::PVChunkTransform();
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	PVRush::PVInput_p ipcap(new PVRush::PVInputPcap(file->path().toLocal8Bit().constData()));
	PVRush::PVChunkAlign* align_org = new PVRush::PVChunkAlign();
	// FIXME: chunk size must be computed somewhere once and for all !
	source_p src = PVRush::PVRawSourceBase_p(new PVRush::PVRawSource<>(ipcap, *align_org, 16000, *transform_null, chk_flt->f()));

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
