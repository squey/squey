#include "PVSourceCreatorTextfile.h"
#include <pvrush/PVInputFile.h>
#include <pvrush/PVNormalizer.h>

#include <pvfilter/PVChunkFilter.h>

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
	
	// So, it always returns true.
	return true;
}

QString PVRush::PVSourceCreatorTextfile::name() const
{
	return QString("text");
}
