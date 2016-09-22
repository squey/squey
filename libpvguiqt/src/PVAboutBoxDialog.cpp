/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>

#include <pvguiqt/PVAboutBoxDialog.h>

#include <License.h>

#include <iostream>
#include <iterator>
#include <sstream>

#include <QApplication>
#include <QGLWidget>
#include <QGridLayout>
#include <QPushButton>

#include <pvguiqt/PVLogoScene.h>

#include <cassert>

PVGuiQt::PVAboutBoxDialog::PVAboutBoxDialog(QWidget* parent /*= 0*/) : QDialog(parent)
{
	setWindowTitle("About INENDI Inspector");

	auto main_layout = new QGridLayout;
	main_layout->setHorizontalSpacing(0);
	main_layout->setSizeConstraint(QLayout::SetFixedSize);

	QString content = "INENDI Inspector version " + QString(INENDI_CURRENT_VERSION_STR) + " \"" +
	                  QString(INENDI_VERSION_NAME) +
	                  "\"<br/>(c) 2014 Picviz Labs SAS, 2015 ESI Group<br/>";

	content += "<br/>contact - <a href=\"mailto:";
	content += EMAIL_ADDRESS_CONTACT;
	content += "?subject=%5BINENDI%5D\">";
	content += EMAIL_ADDRESS_CONTACT;
	content += "</a><br/>";
	content += "support - <a href=\"mailto:";
	content += EMAIL_ADDRESS_SUPPORT;
	content += "?subject=%5BINENDI%5D\">";
	content += EMAIL_ADDRESS_SUPPORT;
	content += "</a><br/>";
	content += "website - <a "
	           "href=\"http://www.esi-inendi.com\">www.esi-inendi.com</a><br/>";

	content += QString("Licence expiration in %1 days<br/>")
	               .arg(Inendi::Utils::License::get_remaining_days(INENDI_FLEX_PREFIX,
	                                                               INENDI_FLEX_FEATURE));

	content += "<br/>With OpenCL support";
	content += "<br/>QT version " + QString(QT_VERSION_STR);

	_view3D_layout = new QHBoxLayout();
	_view3D_layout->setSpacing(0);
	_view3D = new __impl::GraphicsView(this);
	_view3D->setStyleSheet("QGraphicsView { background-color: white; color: "
	                       "white; border-style: none; }");
	_view3D->setViewport(new QGLWidget(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer |
	                                             QGL::SampleBuffers | QGL::DirectRendering)));
	_view3D->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	_view3D->setScene(new PVGuiQt::PVLogoScene());
	_view3D->setCursor(Qt::OpenHandCursor);
	_view3D_layout->addWidget(_view3D);
	auto logo = new QLabel;
	logo->setPixmap(QPixmap(":/logo_text.png"));
	_view3D_layout->addWidget(logo);

	auto text = new QLabel(content);
	text->setAlignment(Qt::AlignCenter);
	text->setTextFormat(Qt::RichText);
	text->setTextInteractionFlags(Qt::TextBrowserInteraction);
	text->setOpenExternalLinks(true);

	QPushButton* ok = new QPushButton("OK");

	assert(DOC_PATH && "The documentation path is not defined.");

	auto doc = new QLabel();
	doc->setText("<br/>Reference Manual: <a href=\"file://" DOC_PATH
	             "/inendi_inspector_reference_manual/index.html\">HTML</a> | "
	             "<a href=\"file://" DOC_PATH
	             "/inendi_inspector_reference_manual.pdf\">PDF</a><br/><br/>"
	             "All documentations: <a href=\"file://" DOC_PATH "/\">local files</a> | "
	             "<a href=\"https://docs.picviz.com\">docs.picviz.com</a><br/><br/>");
	doc->setTextFormat(Qt::RichText);
	doc->setTextInteractionFlags(Qt::TextBrowserInteraction);
	doc->setOpenExternalLinks(true);
	doc->setAlignment(Qt::AlignCenter);

	main_layout->addLayout(_view3D_layout, 0, 0);
	main_layout->addWidget(text, 1, 0);
	main_layout->addWidget(doc, 2, 0);
	main_layout->addWidget(ok, 3, 0);

	setLayout(main_layout);

	connect(ok, SIGNAL(clicked()), this, SLOT(accept()));
}

void PVGuiQt::__impl::GraphicsView::resizeEvent(QResizeEvent* event)
{
	if (scene()) {
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
	}
	QGraphicsView::resizeEvent(event);
}
