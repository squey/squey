#include <QMainWindow>
#include <QApplication>
#include <ctime>
#include <cstdlib>

#include "View.h"
int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	QMainWindow *window = new QMainWindow();
	View *v = new View(window);

	return app.exec();
}
