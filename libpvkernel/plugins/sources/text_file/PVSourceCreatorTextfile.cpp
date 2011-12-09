#include "PVSourceCreatorTextfile.h"


#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNormalizer.h>
#include <pvkernel/rush/PVInputPcap.h>

#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorTextfile::source_p PVRush::PVSourceCreatorTextfile::create_discovery_source_from_input(input_type input, const PVFormat& /*format*/) const
{
	PVLOG_DEBUG("(text_file plugin) create source for %s\n", qPrintable(input->human_name()));
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(file->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src = source_p(new PVRush::PVUnicodeSource<>(ifile, 9000000, chk_flt->f()));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorTextfile::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorTextfile::supported_type() const
{
	return QString("file");
}

bool PVRush::PVSourceCreatorTextfile::pre_discovery(input_type input) const
{
	// AG: I don't know a magic method for being sure that a file is a text-file
	// We'll let the TBB filters work for the moment...
	
	// Just a special case: if this is a pcap, return false
	pcap_t *pcaph;
	char errbuf[PCAP_ERRBUF_SIZE];

	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	pcaph = pcap_open_offline(file->path().toLocal8Bit().constData(), errbuf);
	if (pcaph) {
		pcap_close(pcaph);
		return false;
	}

	return true;
}

QString PVRush::PVSourceCreatorTextfile::name() const
{
	return QString("text");
}
