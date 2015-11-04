/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

