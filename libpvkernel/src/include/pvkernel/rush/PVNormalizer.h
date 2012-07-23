/**
 * \file PVNormalizer.h
 *
 * Copyright (C) Picviz Labs 2010-2012
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
