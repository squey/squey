/**
 * \file vertical_tabs.cpp
 *
 * Copyright (C) Picviz Labs 2012
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

#include <picviz/PVMapped.h>
#include <picviz/PVPlotted.h>
#include <picviz/PVSource.h>
#include <picviz/PVView.h>

#include "common.h"
#include "test-env.h"

#include <iostream>

#include <pvguiqt/PVProjectsTabWidget.h>

class CustomMainWindow : public QMainWindow
{
public:
	CustomMainWindow(QWidget* parent = 0) : QMainWindow(parent)
	{
		setGeometry(
			QStyle::alignedRect(
					Qt::LeftToRight,
					Qt::AlignCenter,
					size(),
					QApplication::desktop()->availableGeometry()
			));
		resize(1024, 768);
	}
};

class PVSplitterHandle : public QSplitterHandle
{
public:
	PVSplitterHandle(Qt::Orientation orientation, QSplitter* parent = 0) : QSplitterHandle(orientation, parent) {}
	void set_max_size(int max_size) { _max_size = max_size; }
protected:
	void mouseMoveEvent(QMouseEvent* event) override
	{
		// assert(_max_size > 0) // set splitter handle max size!
		QList<int> sizes = splitter()->sizes();
		//assert(sizes.size() > 0);
		if ((sizes[0] == 0 && event->pos().x() < _max_size) || (sizes[0] != 0 && event->pos().x() < 0)) {
			QSplitterHandle::mouseMoveEvent(event);
		}
	}
private:
	int _max_size = 0;
};

class PVSplitter : public QSplitter
{
public:
	PVSplitter(Qt::Orientation orientation, QWidget * parent = 0) : QSplitter(orientation, parent) {}

protected:
    QSplitterHandle *createHandle()
    {
    	return new PVSplitterHandle(orientation(), this);
    }
};

int main(int argc, char** argv)
{
	if (argc <= 2) {
		std::cerr << "Usage: " << argv[0] << " file format" << std::endl;
		return 1;
	}

	PVCore::PVIntrinsics::init_cpuid();
	init_env();

	// Get a Picviz tree from the given file/format
	Picviz::PVRoot_p root;
	Picviz::PVSource_sp src = get_src_from_file(root, argv[1], argv[2]);
	Picviz::PVSource_sp src2 = get_src_from_file(root->get_children().at(0), argv[1], argv[2]);
	src2->create_default_view();
	src->create_default_view();


	Picviz::PVView_p view(src->current_view()->get_parent()->shared_from_this());
	view->process_parent_plotted();

	// Qt app
	QApplication app(argc, argv);

	CustomMainWindow* mw = new CustomMainWindow();

	PVGuiQt::PVProjectsTabWidget* projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(root.get(), mw);

	projects_tab_widget->add_source(src.get());

	//projects_tab_widget->collapse_tabs();
	//projects_tab_widget->collapse_tabs(false);

	mw->show();

	return app.exec();
}
