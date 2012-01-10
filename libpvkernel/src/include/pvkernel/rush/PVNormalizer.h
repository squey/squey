/*
 * $Id: PVNormalizer.h 3181 2011-06-21 07:15:22Z aguinet $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVRUSH_PVNORMALIZER_H
#define PVRUSH_PVNORMALIZER_H

#include <QString>
#include <QStringList>

#include <pvkernel/core/general.h>

namespace PVRush {

	QStringList LibKernelDecl normalize_get_plugins_dirs();
	QStringList LibKernelDecl normalize_get_helpers_plugins_dirs(QString helper);

};

#endif	/* PVRUSH_PVNORMALIZER_H */
