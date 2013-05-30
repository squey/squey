
#include <pvkernel/widgets/PVTextPopupWidget.h>

#include <QVBoxLayout>
#include <QWebView>
#include <QWebPage>
#include <QFile>

#define DEFAULT_HTML_TEXT "<html><body style=\"background-color: #1F1F1F; webkit-opacity;0.9; color: white;\">default text</body></html>"

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::PVTextPopupWidget
 *****************************************************************************/

PVWidgets::PVTextPopupWidget::PVTextPopupWidget(QWidget* parent) :
	PVWidgets::PVPopupWidget::PVPopupWidget(parent)
{
	QVBoxLayout* l = new QVBoxLayout();
	l->setContentsMargins(0, 0, 0, 0);
	setLayout(l);

	_webview = new QWebView();
	l->addWidget(_webview);
	_webview->setHtml(DEFAULT_HTML_TEXT);

	// no need for "reload" context menu
	_webview->setContextMenuPolicy(Qt::NoContextMenu);

	setWindowFlags(Qt::Popup | Qt::FramelessWindowHint);
	setWindowOpacity(0.9);
}

/*****************************************************************************
 * PVWidgets::PVTextPopupWidget::setText
 *****************************************************************************/

void PVWidgets::PVTextPopupWidget::setText(const QString& text)
{
	_webview->setHtml(text);
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
		text = "<html><body style=\"background-color: #1F1F1F; webkit-opacity;0.5; color: white;\">ERROR: file <b>" + filename + "</b> can not be loaded</body></html>";
	}

	setText(text);
}
