/**
 * \file PVUtils.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVWIDGETS_PVUTILS_H__
#define __PVWIDGETS_PVUTILS_H__

#include <QFont>
#include <QString>

namespace PVWidgets
{

namespace PVUtils
{
	QString shorten_path(const QString& s, const QFont& font, uint64_t nb_px);
};

}

#endif // __PVWIDGETS_PVUTILS_H__
