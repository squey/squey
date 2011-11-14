#include "PVSourceCreatorPerlfile.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

extern "C" {
#include <EXTERN.h>
#include <perl.h>
}

#define DEFAULT_PERL_CHUNK_SIZE 1024 * 100

PVRush::PVSourceCreatorPerlfile::source_p PVRush::PVSourceCreatorPerlfile::create_discovery_source_from_input(input_type input, const PVFormat& format) const
{
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	source_p src = PVRush::PVPerlSource_p(new PVRush::PVPerlSource(input, DEFAULT_PERL_CHUNK_SIZE, chk_flt->f(), format.get_name()));

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
