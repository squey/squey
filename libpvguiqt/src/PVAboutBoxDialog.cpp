/**
 * \file PVAboutBoxDialog.cpp
 *
 * Copyright (C) Picviz Labs 2012
 */

#include <pvbase/general.h>

#include <pvguiqt/PVAboutBoxDialog.h>

#include <pvkernel/core/PVVersion.h>

#include <iostream>

#include <QApplication>
#include <QGLWidget>
#include <QGridLayout>
#include <QPushButton>

#include <pvguiqt/PVLogoScene.h>

PVGuiQt::PVAboutBoxDialog::PVAboutBoxDialog(QWidget* parent /*= 0*/) : QDialog(parent)
{
	setWindowTitle("About Picviz Inspector");

	QGridLayout *main_layout = new QGridLayout;
	main_layout->setHorizontalSpacing(0);

	QString content = "Picviz Inspector version " + QString(PICVIZ_CURRENT_VERSION_STR) + " \"" + QString(PICVIZ_VERSION_NAME) + "\"<br/>(c) 2010-2014 Picviz Labs SAS<br/>";

	content += "<br/>contact - <a href=\"mailto:contact@picviz.com\">contact@picviz.com</a><br/>";
	content += "support - <a href=\"mailto:support@picviz.com\">support@picviz.com</a><br/>";
	content += "website - <a href=\"http://www.picviz.com\">www.picviz.com</a><br/>";

	content += QString("<br/>Maximum events per source: %L1<br/>").arg(PICVIZ_LINES_MAX);
	content += QString("Licence expiration day: %1<br/>").arg(CUSTOMER_EXPIRATIONDAY);

#ifdef CUDA
	content += "<br/>With CUDA support<br/>";
#endif
	content += "<br/>QT version " + QString(QT_VERSION_STR);

	_view3D_layout = new QHBoxLayout();
	_view3D_layout->setSpacing(0);
	_view3D = new __impl::GraphicsView(this);
	_view3D->setStyleSheet("QGraphicsView { background-color: white; color: white; border-style: none; }");
	_view3D->setViewport(new QGLWidget(QGLFormat(QGL::DoubleBuffer | QGL::DepthBuffer | QGL::SampleBuffers | QGL::DirectRendering)));
	_view3D->setViewportUpdateMode(QGraphicsView::FullViewportUpdate);
	_view3D->setScene(new PVGuiQt::PVLogoScene());
	_view3D->setCursor(Qt::OpenHandCursor);
	_view3D_layout->addWidget(_view3D);
	QLabel *logo = new QLabel;
	logo->setPixmap(QPixmap(":/logo_text.png"));
	_view3D_layout->addWidget(logo);

	QLabel *text = new QLabel(content);
	text->setAlignment(Qt::AlignCenter);
	text->setTextFormat(Qt::RichText);
	text->setTextInteractionFlags(Qt::TextBrowserInteraction);
	text->setOpenExternalLinks(true);

	QPushButton *ok = new QPushButton("OK");

	QLabel* doc = new QLabel();
	doc->setText("<br/>Reference Manual: <a href=\"file:///opt/picviz-inspector/docs/picviz_inspector_reference_manual/index.html\">HTML</a> | " \
	             "<a href=\"file:///opt/picviz-inspector/docs/picviz_inspector_reference_manual.pdf\">PDF</a><br/><br/>" \
	             "All documentations: <a href=\"file:///opt/picviz-inspector/docs/\">local files</a> | " \
	             "<a href=\"https://docs.picviz.com\">docs.picviz.com</a><br/><br/>" \
	);
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

void PVGuiQt::PVAboutBoxDialog::keyPressEvent(QKeyEvent * event)
{
	bool change_mode = false;

	if (event->key() == Qt::Key_F) {
		change_mode = true;
		_fullscreen = !_fullscreen;
	}
	else if (event->key() == Qt::Key_Escape) {
		change_mode = true;
		_fullscreen = false;
	}

	if (change_mode) {
		_view3D->set_fullscreen(_fullscreen);
	}
}

void PVGuiQt::__impl::GraphicsView::keyPressEvent(QKeyEvent* event)
{
	if (_parent->_fullscreen) {
		if (event->key() == Qt::Key_F || event->key() == Qt::Key_Escape) {
			set_fullscreen(false);
		}
	}
	else {
		QGraphicsView::keyPressEvent(event);
	}
}

void PVGuiQt::__impl::GraphicsView::resizeEvent(QResizeEvent *event)
{
	if (scene()) {
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
	}
	QGraphicsView::resizeEvent(event);
}

void PVGuiQt::__impl::GraphicsView::set_fullscreen(bool fullscreen /*= true*/)
{
	// Note: in order to hide the dialog, we have to set it as non modal.
	//       However it seems impossible to set it modal afterwards...
	//       Hiding the dialog works too, but again, unable to show it afterwards!

	if (fullscreen) {
		_parent->setModal(false);
		setParent(0);
		showFullScreen();
	}
	else {
		_parent->_view3D_layout->addWidget(this);
		showNormal();
	}

	_parent->_fullscreen = fullscreen;
}
