/**
 * \file functional_main.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QApplication>

#include "functional_dlg.h"

int main(int argc, char** argv)
{
	QApplication app(argc, argv);

	FunctionalDlg dlg(nullptr);

	dlg.show();

	app.exec();

	return 0;
}
