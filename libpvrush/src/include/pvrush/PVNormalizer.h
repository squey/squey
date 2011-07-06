/*
 * $Id: PVNormalizer.h 3181 2011-06-21 07:15:22Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVRUSH_PVNORMALIZER_H
#define PVRUSH_PVNORMALIZER_H

#include <QLibrary>
#include <QList>
#include <QString>
#include <QStringList>
#include <QHash>
#include <QVector>

#include <pvcore/general.h>

#include <pvrush/PVFormat.h>
#include <pvrush/PVNraw.h>

namespace PVRush {

#ifndef normalize_init_string 
    #define normalize_init_string "picviz_normalize_init"
#endif
#ifndef normalize_file_string 
    #define normalize_file_string "picviz_normalize_file"
#endif
#ifndef normalize_buffer_string 
    #define normalize_buffer_string "picviz_normalize_buffer"
#endif
#ifndef normalize_get_format_string 
    #define normalize_get_format_string "picviz_normalize_get_format"
#endif
#ifndef normalize_terminate_string 
    #define normalize_terminate_string "picviz_normalize_terminate"
#endif
#ifndef normalize_discovery_string 
    #define normalize_discovery_string "picviz_normalize_discovery"
#endif
#ifndef normalize_list_string 
    #define normalize_list_string "picviz_normalize_list"
#endif

typedef void (*normalize_init_function)();
typedef QVector<QStringList> (*normalize_file_function)(PVRush::PVFormat *format, QString filename, QHash<QString, QString> logopt);
typedef QList<QList<QString> > (*normalize_buffer_function)(PVRush::PVFormat format, QString buffer);
typedef int (*normalize_get_format_function)(QString filename, QHash<QString, QString> logopt, PVRush::PVFormat *format);
typedef void (*normalize_terminate_function)(void);

typedef int (*normalize_discovery_function)(QString filename, QStringList *discovered);
typedef int (*normalize_list_function)(QStringList *list);

	QStringList LibExport normalize_get_plugins_dirs();
	QStringList LibExport normalize_get_helpers_plugins_dirs(QString helper);

class NormalizeFunctions {
public:
	QLibrary *lib;	/* We need this to destroy it later */
	normalize_init_function init_function;
	normalize_file_function file_function;
	normalize_buffer_function buffer_function;
	normalize_terminate_function terminate_function;
	normalize_discovery_function discovery_function;
	normalize_get_format_function get_format_function;
	normalize_list_function list_function;
};

/**
 * \class Normalize
 *
 * \defgroup Normalization Input Normalization
 *
 * \brief Normalization means transforming data (pcap, csv, syslog etc.) into a simple table
 *
 * This class loads and run normalization plugins. They are located in
 * the normalize directory that can be defined by PVRUSH_NORMALIZE_DIR
 * environment variable.
 *
 * The normalize plugins often need some extra helpers to work. Such as
 * the pcre normalization (that create an array based on a regex on data
 * such as plaintext logs from applications (syslog, bind, proxy, ..).
 * Those extra plugins are put in the normalize-helpers directory. Also
 * relocatable using the PVRUSH_NORMALIZE_HELPERS_DIR environment variable.
 *
 * Each normalization plugin must implement a set of functions, which are:
 * - normalize_init_function
 * - normalize_file_function
 * - normalize_buffer_function
 * - normalize_get_format_function
 * - normalize_terminate_function
 * - normalize_discover_function
 * - normalize_list_function
 *
 */
class LibExport PVNormalizer {
public:
	PVNormalizer();
	~PVNormalizer();

	/* Methods */
	/**
	* Discover which plugins can be used to a given file
	*
	* The discovery process with run the various tests needed
	* to see which plugins apply to a given filename.
	*
	* @param filename the filename we should run discovery on
	*
	* @return a QStringList of all the plugins that can be run on this file.
	* Empty list if none match.
	*/
	QStringList discover(QString filename);

	/**
	* Normalize a file
	*
	* Creates a list array of normalize data from filename using the
	* type provided.
	*
	* @param type the type to use to run normalization
	* @param filename the filename to be normalized
	*
	* @return nraw of normalized data
	*/
	PVRush::PVNraw *normalize(QString type, QString filename);
	void normalized_debug(QVector<QStringList> qt_nraw);
	QString normalized_get_data(QVector<QStringList> qt_nraw, PVRow i, PVCol j);
	QStringList plugins_list_all();
	int plugins_register_one(QString filename);
	int plugins_register_all();

	/* Attributes */
	QHash<QString, PVRush::NormalizeFunctions> functions;
	PVRush::PVFormat *format;

};
};

#endif	/* PVRUSH_PVNORMALIZER_H */
