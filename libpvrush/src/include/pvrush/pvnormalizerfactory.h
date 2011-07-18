/*
 * $Id: pvnormalizerfactory.h 2502 2011-04-25 18:47:17Z psaade $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVRUSH_NORMALIZERFACTORY_H
#define PVRUSH_NORMALIZERFACTORY_H


#include <QLibrary>
#include <QList>
#include <QString>
#include <QStringList>
#include <QHash>

#include <pvcore/general.h>

#include <pvrush/PVFormat.h>


namespace PVRush {

/**
 * \class PVRush::NormalizerFactory
 *
 * \defgroup Normalization Input Normalization
 *
 * \brief This catches all Normalization possibilities
 *
 * This class loads and detectes all normalizer plugins.
 * They are located in the normalize directory that can be
 * defined by PVRUSH_NORMALIZE_DIR environment variable.
 *
 * The normalize plugins often need some extra helpers to work. Such as
 * the pcre normalization (that create an array based on a regex on data
 * such as plaintext logs from applications (syslog, bind, proxy, ..).
 * Those extra plugins are put in the normalize-helpers directory. Also
 * relocatable using the PVRUSH_NORMALIZE_HELPERS_DIR environment variable.
 *
 */
class LibRushDecl PVNormalizerFactory {

private:
	QStringList normalizer_dirs;

	QStringList get_normalizer_dirs();
	QStringList get_normalizer_helper_dirs(QString helper);

public:
	PVNormalizerFactory();
	~PVNormalizerFactory();

};

// typedef void (*normalize_init_function)();
// typedef QList<QStringList> (*normalize_file_function)(PVRush::Format *format, QString filename, QHash<QString, QString> logopt);
// typedef QList<QList<QString> > (*normalize_buffer_function)(PVRush::Format format, QString buffer);
// typedef PVRush::Format *(*normalize_get_format_function)(QString filename, QHash<QString, QString> logopt);
// typedef void (*normalize_terminate_function)(void);
// 
// typedef QStringList (*normalize_discovery_function)(QString filename);
// typedef QStringList (*normalize_list_function)(void);
// 
// 	QStringList normalize_get_plugins_dirs();
// 	QStringList normalize_get_helpers_plugins_dirs(QString helper);
// 
// 	class NormalizeFunctions {
// 	public:
// 		QLibrary *lib;	/* We need this to destroy it later */
// 		normalize_init_function init_function;
// 		normalize_file_function file_function;
// 		normalize_buffer_function buffer_function;
// 		normalize_terminate_function terminate_function;
// 		normalize_discovery_function discovery_function;
// 		normalize_get_format_function get_format_function;
// 		normalize_list_function list_function;
// 	};
// 
// /**
//  * This is the Normalize class
//  */
// 	class LibRushDecl Normalize {
// 	public:
// 		Normalize();
// 		~Normalize();
// 
// 		/* Methods */
// 		/**
// 		 * Discover which plugins can be used to a given file
// 		 *
// 		 * The discovery process with run the various tests needed
// 		 * to see which plugins apply to a given filename.
// 		 *
// 		 * @param filename the filename we should run discovery on
// 		 *
// 		 * @return a QStringList of all the plugins that can be run on this file.
// 		 * Empty list if none match.
// 		 */
// 		QStringList discover(QString filename);
// 
// 		/**
// 		 * Normalize a file
// 		 *
// 		 * Creates a list array of normalize data from filename using the
// 		 * type provided.
// 		 *
// 		 * @param type the type to use to run normalization
// 		 * @param filename the filename to be normalized
// 		 *
// 		 * @return list array of normalized data
// 		 */
// 		QList<QStringList> normalize(QString type, QString filename);
// 		void normalized_debug(QList<QStringList> qt_nraw);
// 		QString normalized_get_data(QList<QStringList> qt_nraw, PVRow i, PVColumn j);
// 		QStringList plugins_list_all();
// 		int plugins_register_one(QString filename);
// 		int plugins_register_all();
// 
// 		/* Attributes */
// 		QHash<QString, PVRush::NormalizeFunctions> functions;
// 		PVRush::Format *format;
// 
// 	};


};

#endif	/* PVRUSH_NORMALIZERFACTORY_H */

