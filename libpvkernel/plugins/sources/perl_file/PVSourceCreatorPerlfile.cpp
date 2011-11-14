#include "PVSourceCreatorPerlfile.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

#define DEFAULT_PERL_CHUNK_SIZE 1024 * 100

PVRush::PVSourceCreatorPerlfile::source_p PVRush::PVSourceCreatorPerlfile::create_discovery_source_from_input(input_type input, const PVFormat& format) const
{
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	QFileInfo perl_file_info(format.get_full_path());
	QString perl_file(perl_file_info.dir().absoluteFilePath(perl_file_info.completeBaseName() + ".pl"));

	source_p src(new PVRush::PVPerlSource(input, DEFAULT_PERL_CHUNK_SIZE, chk_flt->f(), perl_file));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorPerlfile::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorPerlfile::supported_type() const
{
	return QString("file");
}

bool PVRush::PVSourceCreatorPerlfile::pre_discovery(input_type input) const
{
	return true;
}

QString PVRush::PVSourceCreatorPerlfile::name() const
{
	return QString("perl");
}
