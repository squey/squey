/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef LOGVIEWER_CONFIG_H
#define LOGVIEWER_CONFIG_H

// Allow to show curl debug
#define ADD_CURL_DEBUG 1

// Allow to store debug in file
//#define ADD_DEBUG_TO_FILE 1

// Define temporary template filename
#define TEMPORARYFILENAME_TEMPLATE                                                                 \
	QDir::tempPath() + QLatin1Char('/') + QLatin1String("inendi_temp")

#endif /* LOGVIEWER_CONFIG_H */
