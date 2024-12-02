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

#include <QApplication>
#include <QMainWindow>
#include <QTabWidget>
#include <QStyle>
#include <QWidget>
#include <QSplitter>
#include <QTabBar>
#include <QTextEdit>
#include <QResizeEvent>
#include <QSplitterHandle>
#include <QStackedWidget>
#include <QScreen>

#include <squey/PVMapped.h>
#include <squey/PVScaled.h>
#include <squey/PVSource.h>

#include "common.h"
#include "test-env.h"

#include <iostream>

#include <pvguiqt/PVProjectsTabWidget.h>

class CustomMainWindow : public QMainWindow
{
  public:
	CustomMainWindow(QWidget* parent = nullptr) : QMainWindow(parent)
	{
		setGeometry(QStyle::alignedRect(Qt::LeftToRight, Qt::AlignCenter, size(),
		                                QGuiApplication::screens()[0]->geometry()));
		resize(1024, 768);
	}
};

class PVSplitterHandle : public QSplitterHandle
{
  public:
	PVSplitterHandle(Qt::Orientation orientation, QSplitter* parent = nullptr)
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

	init_env();

	// Get a SQUEY tree from the given file/format
	Squey::PVRoot root;
	Squey::PVSource& src = get_src_from_file(root, argv[1], argv[2]);
	Squey::PVSource& src2 = get_src_from_file(*root.get_children().front(), argv[1], argv[2]);
	src2.emplace_add_child()  // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View

	src.current_view()->get_parent().emplace_add_child();

	// Qt app
	QApplication app(argc, argv);

	auto* mw = new CustomMainWindow();

	auto* projects_tab_widget = new PVGuiQt::PVProjectsTabWidget(&root, mw);

	projects_tab_widget->add_source(&src);

	mw->show();

	return app.exec();
}
