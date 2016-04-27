/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVSourceCreatorPerlfile.h"

#include <pvkernel/rush/PVFileDescription.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

#include <fstream>

#define DEFAULT_PERL_CHUNK_SIZE 1024 * 100

PVRush::PVSourceCreatorPerlfile::source_p
PVRush::PVSourceCreatorPerlfile::create_source_from_input(PVInputDescription_p input,
                                                          const PVFormat& format) const
{
	QFileInfo perl_file_info(format.get_full_path());
	QString perl_file(
	    perl_file_info.dir().absoluteFilePath(perl_file_info.completeBaseName() + ".pl"));

	QFileInfo fi(perl_file);

	if (not fi.exists()) {
		throw std::ifstream::failure("Unknown file, " + perl_file.toStdString() +
		                             " can't create source.");
	}

	return source_p{new PVRush::PVPerlSource(input, DEFAULT_PERL_CHUNK_SIZE, perl_file)};
}

PVRush::hash_formats PVRush::PVSourceCreatorPerlfile::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorPerlfile::supported_type() const
{
	return QString("file");
}

bool PVRush::PVSourceCreatorPerlfile::pre_discovery(PVInputDescription_p /*input*/) const
{
	return true;
}

QString PVRush::PVSourceCreatorPerlfile::name() const
{
	return QString("perl");
}
