/**
 * \file PVSourceCreatorPythonfile.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include "PVPythonSource.h"
#include "PVSourceCreatorPythonfile.h"

#include <pvkernel/rush/PVFileDescription.h>
#include <pvkernel/filter/PVChunkFilter.h>

#include <QDir>
#include <QStringList>
#include <QFileInfo>

#define DEFAULT_PYTHON_CHUNK_SIZE 1024 * 100

PVRush::PVSourceCreatorPythonfile::source_p PVRush::PVSourceCreatorPythonfile::create_discovery_source_from_input(PVInputDescription_p input, const PVFormat& format) const
{
	PVFilter::PVChunkFilter* chk_flt = new PVFilter::PVChunkFilter();
	QFileInfo python_file_info(format.get_full_path());
	QString python_file(python_file_info.dir().absoluteFilePath(python_file_info.completeBaseName() + ".py"));

	source_p src(new PVRush::PVPythonSource(input, DEFAULT_PYTHON_CHUNK_SIZE, chk_flt->f(), python_file));

	return src;
}

PVRush::hash_formats PVRush::PVSourceCreatorPythonfile::get_supported_formats() const
{
	return PVRush::PVFormat::list_formats_in_dir(name(), name());
}

QString PVRush::PVSourceCreatorPythonfile::supported_type() const
{
	return QString("file");
}

bool PVRush::PVSourceCreatorPythonfile::pre_discovery(PVInputDescription_p input) const
{
	return true;
}

QString PVRush::PVSourceCreatorPythonfile::name() const
{
	return QString("python");
}
