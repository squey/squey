/*
 * $Id: PVFileRawSource.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 *
 */

#include <QByteArray>
#include <QFile>
#include <QString>
#include <QStringList>

#include <pvkernel/rush/pv_file_raw_source.h>


PVFileRawSource::PVFileRawSource(const QString &file_name)
{
	filename = QString(file_name);
}


PVFileRawSource::~PVFileRawSource()
{

}


QStringList PVFileRawSource::get_list()
{
	// VARIABLES
	QFile file(filename);
	QStringList output_list;
	QByteArray data_line;

	// CODE
	file.open(QIODevice::ReadOnly | QIODevice::Text);

	data_line = file.readLine();
	while (!data_line.isEmpty()) {
		output_list << QString(data_line);

		data_line = file.readLine();
	}

	file.close();

	return output_list;
}


