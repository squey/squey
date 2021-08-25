//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVUtils.h>
#include <pvbase/general.h>

#include <QApplication>
#include <QDesktopWidget>
#include <QFontMetrics>
#include <QStringList>

#include <math.h>

QString PVWidgets::PVUtils::shorten_path(const QString& s, const QFont& font, uint64_t nb_px)
{
	uint64_t str_width = QFontMetrics(font).horizontalAdvance(s);
	if (str_width < nb_px)
		return s;

	QString str(s);

	const QString separator(INENDI_PATH_SEPARATOR_CHAR);
	const QString elipsis("...");

	QStringList list = str.split(separator);
	uint64_t separator_count = list.length() - 1;
	str_width += QFontMetrics(font).horizontalAdvance(elipsis);

	uint64_t index = 0;
	for (; str_width > nb_px && separator_count > 2; separator_count--) {
		index = (separator_count + 1) / 2;
		str_width -= QFontMetrics(font).horizontalAdvance(list.at(index));
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
		int line_width = QFontMetrics(font).horizontalAdvance(line);
		if (line_width > (int)nb_px) {
			string = string.insert(insert_pos, carriage_return);
			insert_pos += carriage_return.size();
			line.clear();
		} else {
			insert_pos++;
		}
	}
}

uint32_t PVWidgets::PVUtils::tooltip_max_width(QWidget* w)
{
	return QApplication::desktop()->screenGeometry(w).width() / 2;
}

QString PVWidgets::PVUtils::bytes_to_human_readable(size_t byte_count)
{
	QStringList suffix({"B", "KB", "MB", "GB", "TB", "PB", "EB"});
	if (byte_count == 0) {
		return QString::number(0) + suffix[0];
	}
	size_t place = std::floor(std::log(byte_count) / std::log(1024));
	double num = byte_count / std::pow(1024, place);
	return QString::number(num, 'f', 2) + " " + suffix[place];
}
