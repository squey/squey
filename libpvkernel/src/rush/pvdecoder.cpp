/*
 * $Id: pvdecoder.cpp 2531 2011-05-02 20:21:19Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <QLibrary>
#include <QList>
#include <QString>
#include <QStringList>
#include <QRegExp>
#include <QHashIterator>
#include <QHash>
#include <QDir>


#include <stdlib.h>

#include <pvkernel/core/general.h>

#include <pvkernel/rush/PVFormat.h>
#include <pvkernel/rush/pvdecoder.h>

using namespace PVCore;

QStringList PVRush::decoders_get_plugins_dirs()
{
	QString pluginsdirs;
	QStringList pluginsdirs_list; 

	pluginsdirs = QString(getenv("PVRUSH_DECODERS_DIR"));
	if (pluginsdirs.isEmpty()) {
		pluginsdirs = QString(PVRUSH_DECODERS_DIR);
	}

	pluginsdirs_list = pluginsdirs.split(":");
	
	return pluginsdirs_list;
}

PVRush::Decode::Decode()
{
	this->plugins_register_all();
}

PVRush::Decode::~Decode()
{

}

int PVRush::Decode::plugins_register_one(QString filename)
{
	QString plugin_name(filename);
	PVRush::DecodeFunctions nfunctions;


	PVLOG_INFO("Loading decoder plugin: %s\n", filename.toUtf8().data());

	nfunctions.lib = new QLibrary(filename);

	nfunctions.decode_function = (pvrush_decoder_run_function) nfunctions.lib->resolve(pvrush_decoder_run_string);
	if (!nfunctions.decode_function) {
	  PVLOG_ERROR("Error: %s\n", 
		        nfunctions.lib->errorString().toUtf8().data());
	  return -1;
	}

	// From '/foo/bar/libnormalize_pcre.so' to 'pcre'
	plugin_name = plugin_name.section('/', -1);

#ifdef WIN32
	plugin_name.replace(QString("decoder_"), QString(""));
	plugin_name.remove(QRegExp("\\.dll$"));
#else
	plugin_name.replace(QString("libdecoder_"), QString(""));
	plugin_name.remove(QRegExp("\\.so$"));
#endif

	functions[plugin_name] = nfunctions;

	return 0;
}

int PVRush::Decode::plugins_register_all()
{
	QDir dir;
	QStringList pluginsdirs;
	QStringList filters;
	QStringList files;
	int counter;

	pluginsdirs = decoders_get_plugins_dirs();

	for (counter=0; counter < pluginsdirs.count(); counter++) {
		// PVLOG_INFO("Reading decoders plugins directory: %s\n", pluginsdirs[counter].toUtf8().data());
		dir = QDir(pluginsdirs[counter]);

		filters << "*decoder_*";
		dir.setNameFilters(filters);

		files = dir.entryList();
		QStringListIterator filesIterator(files);
		while (filesIterator.hasNext()) {
			plugins_register_one(dir.absoluteFilePath(filesIterator.next()));
		}
	}

	return 0;
}

int PVRush::Decode::decode(PVRush::PVFormat *format, QString decoder, QVector<QStringList> *normalized)
{
	QHash<QString, QString> logopt;
	int retval;

	retval = functions[decoder].decode_function(format, normalized, logopt);

	return retval;
}
