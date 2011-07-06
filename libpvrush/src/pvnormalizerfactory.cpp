/*
 * $Id: pvnormalizerfactory.cpp 2502 2011-04-25 18:47:17Z psaade $
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

#include <pvcore/General>
#include <pvcore/Debug>

#include <pvrush/pvnormalizerfactory.h>


/******************************************************************************
 *
 * PVRush::PVNormalizerFactory::PVNormalizerFactory
 *
 *****************************************************************************/
PVRush::PVNormalizerFactory::PVNormalizerFactory()
{
	normalizer_dirs = get_normalizer_dirs();
	
}

/******************************************************************************
 *
 * PVRush::PVNormalizerFactory::~PVNormalizerFactory
 *
 *****************************************************************************/
PVRush::PVNormalizerFactory::~PVNormalizerFactory()
{

}

/******************************************************************************
 *
 * PVRush::PVNormalizerFactory::get_normalizer_dirs
 *
 *****************************************************************************/
QStringList PVRush::PVNormalizerFactory::get_normalizer_dirs()
{
	QString normalizer_dirs_env_string;
	QStringList normalizer_dirs_list;

	normalizer_dirs_env_string = QString(getenv("PVRUSH_NORMALIZE_DIR"));
	if (normalizer_dirs_env_string.isEmpty()) {
		normalizer_dirs_env_string = QString(PVRUSH_NORMALIZE_DIR);
	}

	normalizer_dirs_list = normalizer_dirs_env_string.split(":");

	return normalizer_dirs_list;
}
