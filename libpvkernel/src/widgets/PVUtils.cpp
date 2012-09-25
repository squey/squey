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
	QString str(s);

	const QString separator(PICVIZ_PATH_SEPARATOR_CHAR);
	const QString eclipsis("...");

	bool shortened = false;
	QStringList list = str.split(separator);
	uint64_t separator_count = list.length()-1;

	for (; (uint64_t) QFontMetrics(font).width(str) > nb_px && separator_count > 2; separator_count--)
	{
		list.removeAt((separator_count+1)/2);

		str = list.join(separator) += eclipsis;
		shortened = true;
	}
	if (shortened) {
		list.insert(separator_count, eclipsis);
		str = list.join(separator);
	}

	return str;
}
