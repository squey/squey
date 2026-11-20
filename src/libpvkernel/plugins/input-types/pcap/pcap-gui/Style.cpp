//
// MIT License
//
// © Squey, 2026
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

#include "Style.h"

#include <pvkernel/core/PVTheme.h>

#include <QDateTime>
#include <QTimeZone>
#include <QLabel>
#include <QLocale>
#include <QPainter>
#include <QPainterPath>
#include <QPalette>
#include <QPushButton>

namespace PVPcapsicum::style
{

namespace
{

bool dark()
{
	return PVCore::PVTheme::is_color_scheme_dark();
}

} // namespace

QColor secondary()
{
	// Between the text and the disabled text of the style sheets of Squey: the palette does
	// not know their colors.
	return dark() ? QColor(0xa7, 0xb1, 0xbb) : QColor(0x6b, 0x74, 0x7d);
}

QColor accent()
{
	return PVCore::PVTheme::link_colors[dark() ? 1 : 0];
}

QColor error()
{
	return dark() ? QColor(0xf2, 0x77, 0x72) : QColor(0xc6, 0x28, 0x28);
}

QColor warning()
{
	return dark() ? QColor(0xe3, 0xb3, 0x41) : QColor(0x9a, 0x67, 0x00);
}

QColor success()
{
	return dark() ? QColor(0x6c, 0xc0, 0x70) : QColor(0x2e, 0x7d, 0x32);
}

QString link(const QString& target, const QString& text)
{
	// The style sheets of Squey leave links in the dark blue of the palette.
	return QString("<a style=\"color: %1;\" href=\"%2\">%3</a>")
	    .arg(accent().name(), target.toHtmlEscaped(), text.toHtmlEscaped());
}

void make_primary(QPushButton* button)
{
	button->setStyleSheet(
	    "QPushButton { background-color: #1a72bb; border: 1px solid #1a72bb; color: #ffffff;"
	    " font-weight: bold; padding: 4px 18px; }"
	    "QPushButton:hover { background-color: #259ae9; border-color: #259ae9; }"
	    "QPushButton:disabled { background-color: transparent; border-color: #788d9c;"
	    " color: #788d9c; }");
}

void color_label(QLabel* label, const QColor& color, bool small)
{
	QPalette palette = label->palette();
	palette.setColor(QPalette::WindowText, color);
	palette.setColor(QPalette::Text, color);
	label->setPalette(palette);
	// The style sheet of Squey sets the color of every widget, which a palette does not.
	label->setStyleSheet(QString("color: %1;%2")
	                         .arg(color.name(), small ? QString(" font-size: small;") : QString()));
}

int paint_tag(
    QPainter& painter, int left, const QRect& line, const QString& text, const QColor& color)
{
	painter.save();
	QFont font = painter.font();
	font.setPointSizeF(font.pointSizeF() * 0.8);
	font.setBold(true);
	painter.setFont(font);
	const QFontMetrics metrics(font);
	const int width = metrics.horizontalAdvance(text) + 10;
	const int height = metrics.height() + 2;
	const QRectF tag(left, line.center().y() - height / 2.0 + 0.5, width, height);
	painter.setRenderHint(QPainter::Antialiasing);
	QColor fill = color;
	fill.setAlphaF(dark() ? 0.18 : 0.12);
	QColor border = color;
	border.setAlphaF(0.55);
	painter.setPen(border);
	painter.setBrush(fill);
	painter.drawRoundedRect(tag, height / 2.0, height / 2.0);
	painter.setPen(color);
	painter.drawText(tag, Qt::AlignCenter, text);
	painter.restore();
	return width;
}

QString size_text(uint64_t bytes)
{
	return QLocale(QLocale::English).formattedDataSize(qint64(bytes), 1, QLocale::DataSizeSIFormat);
}

QString count_text(uint64_t count)
{
	return QLocale(QLocale::English).toString(qulonglong(count));
}

QString short_count_text(uint64_t count)
{
	const QLocale english(QLocale::English);
	if (count >= 1'000'000'000) {
		return english.toString(count / 1e9, 'f', 1) + " G";
	}
	if (count >= 1'000'000) {
		return english.toString(count / 1e6, 'f', 1) + " M";
	}
	if (count >= 10'000) {
		return english.toString(count / 1e3, 'f', 0) + " K";
	}
	return count_text(count);
}

QString time_text(int64_t nanoseconds)
{
	return QDateTime::fromMSecsSinceEpoch(nanoseconds / 1'000'000, QTimeZone::UTC)
	    .toString("yyyy-MM-dd HH:mm");
}

QString type_text(std::string_view type)
{
	if (type == "FT_ABSOLUTE_TIME") {
		return "time";
	}
	if (type == "FT_RELATIVE_TIME") {
		return "seconds";
	}
	if (type == "FT_FRAMENUM") {
		return "frame";
	}
	if (type == "FT_IPv4") {
		return "IPv4";
	}
	if (type == "FT_IPv6") {
		return "IPv6";
	}
	if (type == "FT_ETHER") {
		return "MAC";
	}
	if (type == "FT_BOOLEAN") {
		return "boolean";
	}
	if (type == "FT_FLOAT" or type == "FT_DOUBLE") {
		return "decimal";
	}
	if (type.starts_with("FT_UINT") or type.starts_with("FT_INT")) {
		return type.starts_with("FT_UINT_") ? "text" : "integer";
	}
	return "text";
}

QString to_qstring(std::string_view text)
{
	return QString::fromUtf8(text.data(), qsizetype(text.size()));
}

} // namespace PVPcapsicum::style
