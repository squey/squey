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

#include <pvkernel/widgets/PVImportWidget.h>

#include <QFileInfo>
#include <QFont>
#include <QLabel>
#include <QStyleOptionToolButton>
#include <QStylePainter>
#include <QToolButton>

#include <stdexcept>

PVRush::PVFormat PVWidgets::PVImportWidget::format_of(QString const& path)
{
	if (not QFileInfo(path).isReadable()) {
		throw std::runtime_error(tr("The format '%1' cannot be read.").arg(path).toStdString());
	}
	return {path, path};
}

QLabel* PVWidgets::PVImportWidget::section_title(QString const& text)
{
	auto* title = new QLabel(text.toUpper());
	QFont font = title->font();
	font.setBold(true);
	font.setPointSizeF(font.pointSizeF() * 0.85);
	font.setLetterSpacing(QFont::AbsoluteSpacing, 1);
	title->setFont(font);
	title->setProperty("secondary", true);
	return title;
}

namespace
{

/** Its icon above its text, both in its middle however tall it is. */
class DropZone : public QToolButton
{
  protected:
	void paintEvent(QPaintEvent*) override
	{
		// The edge of the theme alone: a tool button puts its icon at its top, and its text in the
		// middle of what is left.
		QStylePainter painter(this);
		QStyleOptionToolButton option;
		initStyleOption(&option);
		const QIcon icon = option.icon;
		const QString label = option.text;
		option.icon = QIcon();
		option.text.clear();
		painter.drawComplexControl(QStyle::CC_ToolButton, option);

		const int spacing = fontMetrics().height() / 2;
		const QRect text_size = fontMetrics().boundingRect(rect(), Qt::AlignCenter, label);
		const int top = (height() - iconSize().height() - spacing - text_size.height()) / 2;
		const QRect icon_area(QPoint((width() - iconSize().width()) / 2, top), iconSize());
		icon.paint(&painter, icon_area, Qt::AlignCenter,
		           isEnabled() ? QIcon::Normal : QIcon::Disabled);
		style()->drawItemText(
		    &painter, QRect(0, icon_area.bottom() + 1 + spacing, width(), text_size.height()),
		    Qt::AlignHCenter | Qt::AlignTop, palette(), isEnabled(), label, QPalette::ButtonText);
	}
};

} // namespace

QToolButton* PVWidgets::PVImportWidget::drop_zone(QIcon const& icon, QString const& text)
{
	// The theme gives it a dashed edge, by its name.
	auto* zone = new DropZone;
	zone->setObjectName("import_ask_files");
	zone->setIcon(icon);
	zone->setIconSize(QSize(40, 40));
	zone->setText(text);
	zone->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
	zone->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
	zone->setMinimumHeight(8 * zone->fontMetrics().lineSpacing());
	zone->setCursor(Qt::PointingHandCursor);
	return zone;
}
