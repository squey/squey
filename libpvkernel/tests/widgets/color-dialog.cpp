#include <pvkernel/widgets/PVColorDialog.h>
#include <QApplication>

#include <QMainWindow>

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	PVWidgets::PVColorDialog* cp = new PVWidgets::PVColorDialog();
	cp->show();

	return app.exec();
}
