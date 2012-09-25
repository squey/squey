/**
 * \file PVUtils.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/widgets/PVUtils.h>
#include <pvbase/general.h>

#include <QFontMetrics>
#include <QStringList>

QString PVWidgets::PVUtils::shorten_path(const QString& s, const QFont& font, uint64_t nb_px)
{
	uint64_t str_width = QFontMetrics(font).width(s);
	if (str_width < nb_px) return s;

	QString str(s);

	const QString separator(PICVIZ_PATH_SEPARATOR_CHAR);
	const QString elipsis("...");

	QStringList list = str.split(separator);
	uint64_t separator_count = list.length()-1;
	str_width += QFontMetrics(font).width(elipsis);

	uint64_t index;
	for (; str_width > nb_px && separator_count > 2; separator_count--)
	{
		index = (separator_count+1)/2;
		str_width -= QFontMetrics(font).width(list.at(index));
		list.removeAt(index);
	}
	list.insert(index, elipsis);
	str = list.join(separator);

	return str;
}
