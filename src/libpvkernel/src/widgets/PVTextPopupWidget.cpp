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

#include <pvkernel/core/PVTheme.h>
#include <pvkernel/core/PVLogger.h> // for PVLOG_WARN
#include <QtCore/qobjectdefs.h>
#include <qcoreevent.h>
#include <qiodevice.h>
#include <qnamespace.h>
#include <qobject.h>
#include <qwidget.h>
#include <QVBoxLayout>
#include <QByteArray>
#include <QKeyEvent>
#include <QFile>
#include <QString>
#include <QTextBrowser>

#include "pvkernel/widgets/PVPopupWidget.h"     // for PVPopupWidget
#include "pvkernel/widgets/PVTextPopupWidget.h" // for PVTextPopupWidget, etc

/**
 * RH: the code snipset from http://jsfiddle.net/r9yrM/1/ has been used as a template for the
 * documentation which also uses div tags with CSS's display set to table or table-cell.
 */

#define AlignHoriMask                                                                              \
	(PVWidgets::PVTextPopupWidget::AlignLeft | PVWidgets::PVTextPopupWidget::AlignRight |          \
	 PVWidgets::PVTextPopupWidget::AlignHCenter)
#define AlignVertMask                                                                              \
	(PVWidgets::PVTextPopupWidget::AlignTop | PVWidgets::PVTextPopupWidget::AlignBottom |          \
	 PVWidgets::PVTextPopupWidget::AlignVCenter)

//#define DEFAULT_HTML_TEXT "<html><body style=\"background-color: transparent; -webkit-opacity:
// 0.5; color: white;\">default text</body></html>"
#define DEFAULT_HTML_TEXT                                                                          \
	"<html><body style=\"background-color: #FF0000; color: white;\">default text</body></html>"

static void write_open_table(QString& text)
{
	text += "<div>\n";
	text += "<div>\n";
}

static void write_close_table(QString& text)
{
	text += "</div>\n";
	text += "</div>\n";
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::PVTextPopupWidget
 *****************************************************************************/

PVWidgets::PVTextPopupWidget::PVTextPopupWidget(QWidget* parent)
    : PVWidgets::PVPopupWidget::PVPopupWidget(parent), _last_widget(nullptr)
{
	setWindowFlags(Qt::FramelessWindowHint);

	setFocusPolicy(Qt::NoFocus);

	auto l = new QVBoxLayout();
	l->setContentsMargins(0, 0, 0, 0);
	setLayout(l);

	_webview = new QTextBrowser();

	// // no need for "reload" context menu
	// _webview->setContextMenuPolicy(Qt::NoContextMenu);
	_webview->setText(DEFAULT_HTML_TEXT);
	l->addWidget(_webview);

	connect(&PVCore::PVTheme::get(), &PVCore::PVTheme::color_scheme_changed, this, &PVWidgets::PVTextPopupWidget::refresh_theme);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::refresh_theme
 *****************************************************************************/
void PVWidgets::PVTextPopupWidget::refresh_theme(PVCore::PVTheme::EColorScheme /*cs*/)
{
    setText(_temp_text.arg(get_style()));
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::get_style
 *****************************************************************************/

QString PVWidgets::PVTextPopupWidget::get_style()
{
	const QString& css_filename = PVCore::PVTheme::is_color_scheme_light() ? ":help-style-light" : ":help-style-dark";

	QFile file(css_filename);
	QString text;

	QString style;

	if (file.open(QIODevice::ReadOnly)) {
		QByteArray data;
		data = file.read(file.size());
		style += QString(data);
	} else {
		PVLOG_WARN("ignoring help content from '%s' because it can not be loaded\n",
		           qPrintable(css_filename));
	}

	return style;
}
/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setText
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setText(const QString& text)
{
	_webview->setText(text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setTextFromFile
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setTextFromFile(const QString& filename)
{
	QFile file(filename);
	QString text(DEFAULT_HTML_TEXT);

	if (file.open(QIODevice::ReadOnly)) {
		QByteArray data;
		data = file.read(file.size());
		text = QString(data);
	} else {
		text = "<html><body style=\"background-color: #1F1F1F; webkit-opacity: 0.5; color: "
		       "white;\">ERROR: file <b>" +
		       filename + "</b> can not be loaded</body></html>";
	}

	setText(text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::initTextFromFile
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::initTextFromFile(const QString& title)
{
	_temp_text = QString();
	_temp_text += "<html>\n<head>\n<title>" + title + "</title>\n";
	_temp_text += "<style>%1</style>";
	_temp_text += "</head>\n";
	_temp_text += "<body>\n";

	write_open_table(_temp_text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::addTextFromFile
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::addTextFromFile(const QString& html_filename)
{
	QFile file(html_filename);
	QString text;

	if (file.open(QIODevice::ReadOnly)) {
		QByteArray data;
		data = file.read(file.size());
		text += QString(data) + "\n";
	} else {
		PVLOG_WARN("ignoring help content from '%s' because it can not be loaded\n",
		           qPrintable(html_filename));
	}

	_temp_text += text;
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::newColumn
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::newColumn()
{
	_temp_text += "</div>\n<div>\n";
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::newTable
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::newTable()
{
	write_close_table(_temp_text);
	_temp_text += "<br>\n";
	write_open_table(_temp_text);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::finalizeText
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::finalizeText()
{
	write_close_table(_temp_text);
	_temp_text += "</body>\n</html>";
	setText(_temp_text.arg(get_style()));

#if 1
	//RH: may be usefull to dump the constructed
	QFile file("aa.html");
	if (file.open(QIODevice::WriteOnly)) {
		file.write(_temp_text.arg(get_style()).toLocal8Bit());
	}
	file.close();
#endif
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::popup
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::popup(QWidget* widget, int align, int expand, int border)
{
	if (isVisible()) {
		return;
	}

	_last_widget = widget;
	_last_align = align;
	_last_expand = expand;
	_last_border = border;

	// make sure the popup's geometry is correct
	adjustSize();

	setGeometry(parentWidget()->rect());
	raise();
	show();
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setVisible
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setVisible(bool visible)
{
	QWidget::setVisible(visible);

	if (visible) {
		parentWidget()->installEventFilter(this);
	} else {
		parentWidget()->removeEventFilter(this);
		_last_widget = nullptr;
	}

	if (_last_widget) {
		_last_widget->setFocus();
	}
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::eventFilter
 *****************************************************************************/

bool PVWidgets::PVTextPopupWidget::eventFilter(QObject* obj, QEvent* event)
{
	if (_last_widget == nullptr) {
		return false;
	}

	if (event->type() == QEvent::Resize) {
		setGeometry(parentWidget()->rect());

		return false;
	} else if (event->type() == QEvent::KeyPress) {
		if (isVisible()) {
			int key = static_cast<QKeyEvent*>(event)->key();
			if (is_close_key(key)) {
				hide();
				return true;
			}
		}
	}
	// standard event processing
	return QObject::eventFilter(obj, event);
}
