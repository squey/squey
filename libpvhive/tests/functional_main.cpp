
#include <QApplication>

#include "functional_dlg.h"

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	BigTestDlg dlg(nullptr);

	dlg.show();

	app.exec();

	return 0;
}
