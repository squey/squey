/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QWidget>
#include <QDesktopWidget>
#include <QSplitter>
#include <QTabBar>
#include <QTextEdit>
#include <QResizeEvent>
#include <QSplitterHandle>
#include <QStackedWidget>
#include <QScreen>

#include <inendi/PVMapped.h>
#include <inendi/PVPlotted.h>
#include <inendi/PVSource.h>

#include "common.h"
#include "test-env.h"

#include <iostream>

#include <pvguiqt/PVProjectsTabWidget.h>

class CustomMainWindow : public QMainWindow
{
  public:
	CustomMainWindow(QWidget* parent = 0) : QMainWindow(parent)
	{
		setGeometry(QStyle::alignedRect(Qt::LeftToRight, Qt::AlignCenter, size(),
		                                QGuiApplication::screens()[0]->geometry()));
		resize(1024, 768);
	}
};

class PVSplitterHandle : public QSplitterHandle
{
  public:
	PVSplitterHandle(Qt::Orientation orientation, QSplitter* parent = 0)
	    : QSplitterHandle(orientation, parent)
	{
	}
	void set_max_size(int max_size) { _max_size = max_size; }

  protected:
	void mouseMoveEvent(QMouseEvent* event) override
	{
		// assert(_max_size > 0) // set splitter handle max size!
		QList<int> sizes = splitter()->sizes();
		// assert(sizes.size() > 0);
		if ((sizes[0] == 0 && event->pos().x() < _max_size) ||
		    (sizes[0] != 0 && event->pos().x() < 0)) {
			QSplitterHandle::mouseMoveEvent(event);
		}
	}

  private:
	int _max_size = 0;
};

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	PVCore::PVIntrinsics::init_cpuid();
	init_env();

	// Get a INENDI tree from the given file/format
	Inendi::PVRoot root;
	Inendi::PVSource& src = get_src_from_file(root, argv[1], argv[2]);
	Inendi::PVSource& src2 = get_src_from_file(*root.get_children().front(), argv[1], argv[2]);
	src2.emplace_add_child()  // Mapped
	    .emplace_add_child()  // Plotted
	    .emplace_add_child(); // View
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Plotted
	    .emplace_add_child(); // View

	src.current_view()->get_parent().emplace_add_child();

	// Qt app
	QApplication app(argc, argv);

	CustomMainWindow* mw = new CustomMainWindow();

	PVGuiQt::PVProjectsTabWidget* projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(&root, mw);

	projects_tab_widget->add_source(&src);

	mw->show();

	return app.exec();
}
