/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVSourceCreatorTextfile.h"

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/rush/PVInputFile.h>
#include "PVTextFileSource.h"

#include <QDir>
#include <QStringList>
#include <QFileInfo>

PVRush::PVSourceCreatorTextfile::source_p
PVRush::PVSourceCreatorTextfile::create_source_from_input(PVInputDescription_p input) const
{
	QSettings& pvconfig = PVCore::PVConfig::get().config();

	PVLOG_DEBUG("(text_file plugin) create source for %s\n", qPrintable(input->human_name()));
	PVFileDescription* file = dynamic_cast<PVFileDescription*>(input.get());
	assert(file);
	PVRush::PVInput_p ifile(new PVRush::PVInputFile(file->path().toLocal8Bit().constData()));
	// FIXME: chunk size must be computed somewhere once and for all !
	int size_chunk = pvconfig.value("pvkernel/max_size_chunk").toInt();
	if (size_chunk <= 0) {
		size_chunk = 4096 * 100; // Aligned on a page boundary (4ko)
	}
	source_p src = source_p(new PVTextFileSource(ifile, file, size_chunk));

	return src;
}

QString PVRush::PVSourceCreatorTextfile::supported_type() const
{
	return QString("file");
}

QString PVRush::PVSourceCreatorTextfile::name() const
{
	return QString("text");
}
