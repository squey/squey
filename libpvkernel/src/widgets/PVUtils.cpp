/**
 * \file PVUtils.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvkernel/widgets/PVUtils.h>
#include <pvbase/general.h>

#include <QApplication>
#include <QDesktopWidget>
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

	uint64_t index = 0;
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

void PVWidgets::PVUtils::html_word_wrap_text(QString& string, const QFont& font, uint64_t nb_px)
{
	static const QString carriage_return("<br>");

	int insert_pos = 0;
	QString line;
	while (insert_pos < string.size()) {
		line += string[insert_pos];
		int line_width = QFontMetrics(font).width(line);
		if (line_width > (int) nb_px) {
			string = string.insert(insert_pos, carriage_return);
			insert_pos += carriage_return.size();
			line.clear();
		}
		else {
			insert_pos++;
		}
	}
}

uint32_t PVWidgets::PVUtils::tooltip_max_width(QWidget* w)
{
	return QApplication::desktop()->screenGeometry(w).width()/2;
}
