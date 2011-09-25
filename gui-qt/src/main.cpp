//! \file main.cpp
//! $Id: main.cpp 3191 2011-06-23 13:47:36Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

//! [0]
#include <QApplication>
#include <QLocale>
#include <QTextCodec>
#include <QResource>

#include <PVMainWindow.h>
#include <PVCustomStyle.h>

#include <stdio.h>

#include <time.h>
#include <pvkernel/core/picviz_intrin.h>

#define JULY_5 1309856400

// Laposte
#define SEPT_20 1316520000

// #ifdef USE_UNIKEY
  // #include <UniKeyFR.h>
// #endif


int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	//app.setStyle(new PVInspector::PVCustomStyle());
	PVInspector::PVMainWindow pv_main_window;
	QString wintitle;
	
	// Here, we assume that everyone is coding with an editor using UTF-8
	QTextCodec::setCodecForCStrings(QTextCodec::codecForName("UTF-8"));

// #ifdef USE_UNIKEY
	// DWORD retcode, lp1, lp2;
	// WORD handle[16], p1, p2;

	// p1=65143;
	// p2=39181;

	// retcode = UniKey_Find(&handle[0], &lp1, &lp2);
	// if (retcode) {
	// 	PVLOG_ERROR("Cannot find Unikey. Error code:%d\n", retcode);
	// 	exit(1);
	// }

	// retcode = UniKey_User_Logon(&handle[0], &p1, &p2);
	// if (retcode) {
	// 	PVLOG_ERROR("Logon error. Invalid key?. Error code:%d\n", retcode);
	// 	exit(1);
	// }
// #endif

	time_t t = time(NULL);
 	// PVLOG_INFO("Current time:%d\n", t);
#if 0
	if (t > SEPT_20) {
		exit(42);
	}
#endif

	QString locale = QLocale::system().name();
	PVLOG_INFO("System locale: %s\n", qPrintable(locale));

	PVCore::PVIntrinsics::init_cpuid();
#ifdef __SSE4_1__
	PVLOG_INFO("Compiled with SSE 4.1 instructions\n");
	if (PVCore::PVIntrinsics::has_sse41()) {
		PVLOG_INFO("SSE4.1 is supported by this CPU.\n");
	}
	else {
		PVLOG_INFO("SSE4.1 is not supported by this CPU.\n");
	}
#endif

	app.setOrganizationName("PICVIZ Labs");
	app.setApplicationName("Picviz Inspector");
	app.setWindowIcon(QIcon(":/window-icon.png"));

	QResource res_css(":/gui.css");
	app.setStyleSheet(QString((const char *)res_css.data()));

	wintitle = QString("Picviz Inspector ") + QString(PICVIZ_CURRENT_VERSION_STR);
	pv_main_window.setWindowTitle(wintitle);

	pv_main_window.show();
	int ret = app.exec();

	return ret;
}
//! [0]
