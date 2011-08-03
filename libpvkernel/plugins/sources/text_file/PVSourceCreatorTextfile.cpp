#include "PVSourceCreatorTextfile.h"
#include <pvkernel/rush/PVInputFile.h>
#include <pvkernel/rush/PVNormalizer.h>
#include <pvkernel/rush/PVInputPcap.h>

#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorTextfile::source_p PVRush::PVSourceCreatorTextfile::create_source_from_input(PVCore::PVArgument const& input) const
{
	PVLOG_DEBUG("(text_file plugin) create source for %s\n", qPrintable(input.toString()));
	// input is a QString !
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(input.toString().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src = source_p(new PVRush::PVUnicodeSource<>(ifile, 16000, chk_flt->f()));

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

bool PVRush::PVSourceCreatorTextfile::pre_discovery(PVCore::PVArgument const& input) const
{
	// AG: I don't know a magic method for being sure that a file is a text-file
	// We'll let the TBB filters work for the moment...
	
	// Just a special case: if this is a pcap, return false
	pcap_t *pcaph;
	char errbuf[PCAP_ERRBUF_SIZE];

	pcaph = pcap_open_offline(input.toString().toLocal8Bit().constData(), errbuf);
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
