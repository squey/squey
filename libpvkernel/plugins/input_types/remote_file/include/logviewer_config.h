/**
 * \file logviewer_config.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef LOGVIEWER_CONFIG_H
#define LOGVIEWER_CONFIG_H


//Allow to show curl debug
#define ADD_CURL_DEBUG 1

//Allow to store debug in file
//#define ADD_DEBUG_TO_FILE 1

//Define temporary template filename
#define TEMPORARYFILENAME_TEMPLATE QDir::tempPath() + QLatin1Char( '/' ) +QLatin1String( "picviz_temp" )

#endif /* LOGVIEWER_CONFIG_H */

