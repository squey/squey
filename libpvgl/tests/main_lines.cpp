#include <QMainWindow>
#include <QApplication>
#include <ctime>
#include <cstdlib>

#include <common/common.h>
#include <gl/view.h>

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	QMainWindow *window = new QMainWindow();
	
	View *v = new View(window);
	
	window->setCentralWidget(v);

	window->resize(QSize(1024,1024));
	//v->resize(v->sizeHint());

	Point* buf = allocate_buffer(NB_LINES);
	fill_buffer(buf, NB_LINES);
	v->set_buffer(buf, NB_LINES);

	window->show();
	return app.exec();
}
