//! \file main.cpp
//! $Id: main.cpp 3191 2011-06-23 13:47:36Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

//! [0]
#include <QApplication>
#include <QLocale>
#include <QTextCodec>

#include <PVMainWindow.h>

#include <stdio.h>

#include <time.h>

#define JULY_5 1309856400

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	PVInspector::PVMainWindow pv_main_window;
	QString wintitle;
	
	// Here, we assume that everyone is coding with an editor using UTF-8
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));

	// time_t t = time(NULL);
 	// PVLOG_INFO("Current time:%d\n", t);
	// if (t>JULY_5) {
	// 	exit(42);
	// }

	QString locale = QLocale::system().name();
	PVLOG_INFO("System locale: %s\n", qPrintable(locale));

	app.setOrganizationName("PICVIZ Labs");
	app.setApplicationName("Picviz Inspector");
	app.setWindowIcon(QIcon(":/window-icon.png"));

	wintitle = QString("Picviz Inspector ") + QString(PICVIZ_VERSION_STR);
	pv_main_window.setWindowTitle(wintitle);

	pv_main_window.show();
	int ret = app.exec();

	return ret;
}
//! [0]
