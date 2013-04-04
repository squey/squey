/**
 * \file PVUtils.h
 *
 * Copyright (C) Picviz Labs 2012
 */

#ifndef __PVWIDGETS_PVUTILS_H__
#define __PVWIDGETS_PVUTILS_H__

#include <QFont>
#include <QString>


#define PICVIZ_TOOLTIP_MAX_WIDTH 800

namespace PVWidgets
{

namespace PVUtils
{
	QString shorten_path(const QString& s, const QFont& font, uint64_t nb_px);
	void html_word_wrap_text(QString& string, const QFont& font, uint64_t nb_px);

	/*! \brief Returns the maximum tooltip width related to a widget
	 */
	uint32_t tooltip_max_width(QWidget* w);
};

}

#endif // __PVWIDGETS_PVUTILS_H__
