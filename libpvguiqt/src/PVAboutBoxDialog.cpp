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
#include <QTabWidget>
#include <QListWidget>
#include <QDirIterator>
#include <QTextEdit>
#include <QTextStream>

#include <pvguiqt/PVLogoScene.h>

#include <cassert>

static QString copying_dir()
{
	const char* path = getenv("COPYING_DIR");
	if (path) {
		return path;
	}
	return INENDI_COPYING_DIR;
}

class PVOpenSourceSoftwareWidget : public QWidget
{
  public:
	PVOpenSourceSoftwareWidget(QWidget* parent = nullptr) : QWidget(parent)
	{
		QListWidget* oss_software_list = new QListWidget;

		QDirIterator dir_it(copying_dir(), QDir::Files);
		while (dir_it.hasNext()) {
			oss_software_list->addItem(QFileInfo(dir_it.next()).fileName());
		}
		oss_software_list->sortItems();
		oss_software_list->setMaximumWidth(oss_software_list->sizeHintForColumn(0) + 2);

		QTextEdit* license_text = new QTextEdit;
		license_text->setReadOnly(true);

		QHBoxLayout* layout = new QHBoxLayout;

		layout->addWidget(oss_software_list);
		layout->addWidget(license_text);

		connect(oss_software_list, &QListWidget::currentRowChanged, [=]() {
			const QString& file_path =
			    copying_dir() + "/" + oss_software_list->currentItem()->text();
			QFile f(file_path);
			f.open(QFile::ReadOnly | QFile::Text);
			QTextStream in(&f);
			license_text->setText(in.readAll());
		});

		oss_software_list->setCurrentRow(0);

		setLayout(layout);
	}
};

PVGuiQt::PVAboutBoxDialog::PVAboutBoxDialog(QWidget* parent /*= 0*/) : QDialog(parent)
{
	setWindowTitle("About INENDI Inspector");

	auto main_layout = new QGridLayout;
	main_layout->setHorizontalSpacing(0);

	QString content = "INENDI Inspector version " + QString(INENDI_CURRENT_VERSION_STR) + " \"" +
	                  QString(INENDI_VERSION_NAME) + "\"<br/>© 2015 Picviz Labs SAS<br/>© 2015-" +
	                  QString::number(QDate::currentDate().year()) + " ESI Group<br/>";

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
	           "href=\"http://www.esi-inendi.com\">www.esi-inendi.com</a><br/><br/>";

	content += QString("Licence expires in %1 days.<br/>")
	               .arg(Inendi::Utils::License::get_remaining_days(INENDI_LICENSE_PREFIX,
	                                                               INENDI_LICENSE_FEATURE));

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
	_view3D->setVisible(not getenv("SSH_CLIENT"));
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
	             "/inendi_inspector_reference_manual.pdf\">PDF</a><br/><br/>");
	doc->setTextFormat(Qt::RichText);
	doc->setTextInteractionFlags(Qt::TextBrowserInteraction);
	doc->setOpenExternalLinks(true);
	doc->setAlignment(Qt::AlignCenter);

	QGridLayout* software_layout = new QGridLayout;
	software_layout->setHorizontalSpacing(0);
	software_layout->addLayout(_view3D_layout, 0, 0);
	software_layout->addWidget(text, 1, 0);
	software_layout->addWidget(doc, 2, 0);

	QWidget* tab_software = new QWidget;
	tab_software->setLayout(software_layout);

	QTabWidget* tab_widget = new QTabWidget();
	tab_widget->addTab(tab_software, "Software");
	tab_widget->addTab(new PVOpenSourceSoftwareWidget, "Open source software");

	main_layout->addWidget(tab_widget, 0, 0);
	main_layout->addWidget(ok, 1, 0);

	setLayout(main_layout);

	connect(ok, &QAbstractButton::clicked, this, &QDialog::accept);
}

void PVGuiQt::__impl::GraphicsView::resizeEvent(QResizeEvent* event)
{
	if (scene()) {
		scene()->setSceneRect(QRect(QPoint(0, 0), event->size()));
	}
	QGraphicsView::resizeEvent(event);
}
