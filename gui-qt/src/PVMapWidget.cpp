#if 0
#include <QWebView>

#include <PVMapWidget.h>

#include <picviz/general.h>

PVMapWidget::PVMapWidget(QWidget *parent) : QWebView(parent)
{
	QWebView *view = new QWebView(parent);

	view->load(QUrl("http://www.picviz.com/"));
	view->show();
}
#endif

