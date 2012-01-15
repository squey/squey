//! \file main.cpp
//! $Id: main.cpp 3191 2011-06-23 13:47:36Z stricaud $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

//! [0]
#include <QApplication>
#include <QFile>
#include <QLocale>
#include <QResource>
#include <QString>
#include <QTextStream>
#include <QTextCodec>

#include <PVMainWindow.h>
#include <PVCustomStyle.h>

#include <string>
#include <vector>
#include <iostream>

#include <stdio.h>
//#include <dlfcn.h>

#include <time.h>
#include <pvkernel/core/picviz_intrin.h>

#include <boost/program_options.hpp>

#define JULY_5 1309856400

// #ifdef USE_UNIKEY
  // #include <UniKeyFR.h>
// #endif

namespace bpo = boost::program_options;

int main(int argc, char *argv[])
{
	//dlopen("/lib/x86_64-linux-gnu/libgcc_s.so.1", RTLD_LAZY);
	QApplication app(argc, argv);

	// Program options
	bpo::options_description desc_opts("Options");
	desc_opts.add_options()
		("help", "produce help message")
		("format", bpo::value<std::string>(), "path to the format to use. Default is automatic discovery.")
		("project", bpo::value<std::string>(), "path to the project to load.")
		;
	bpo::options_description hidden_opts("Hidden options");
	hidden_opts.add_options()
		("input-file", bpo::value<std::vector<std::string> >(), "path to the file to load")
		;
	bpo::options_description all_opts;
	all_opts.add(desc_opts).add(hidden_opts);

	bpo::positional_options_description p;
	p.add("input-file", -1);

	bpo::variables_map vm;
	bpo::store(bpo::command_line_parser(argc, argv).options(all_opts).positional(p).run(), vm);
	bpo::notify(vm);

	if (vm.count("help")) {
		std::cerr << "Picviz Inspector " << PICVIZ_CURRENT_VERSION_STR << std::endl << std::endl;
		std::cerr << "Usage: " << argv[0] << " [--format format] [file [file...]]" << std::endl;
		std::cerr << desc_opts << std::endl;
		return 1;
	}

	QString format;
	if (vm.count("format")) {
		std::string format_arg = vm["format"].as<std::string>();
		format = QString::fromLocal8Bit(format_arg.c_str(), format_arg.size());
	}

	std::vector<QString> files;
	if (vm.count("input-file")) {
		std::vector<std::string> files_arg = vm["input-file"].as<std::vector<std::string> >();
		files.reserve(files_arg.size());
		std::vector<std::string>::const_iterator it;
		// Convert file path to unicode
		for (it = files_arg.begin(); it != files_arg.end(); it++) {
			files.push_back(QString::fromLocal8Bit(it->c_str(), it->size()));
		}
	}


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
	app.setApplicationName("Picviz Inspector " PICVIZ_CURRENT_VERSION_STR);
	app.setWindowIcon(QIcon(":/window-icon.png"));


	// We get the CSS as a QFile
	// WARNING: The following line should not be removed. It is for testing conveniences.
	// It is commented by default (PhS)
	//QFile css_file("/donnees/GIT/OLD/picviz-inspector/gui-qt/src/resources/gui.css");
	// This is the real definition
	QFile css_file(":/gui.css");
	
	// We open the QFile
	css_file.open(QFile::ReadOnly);
	// We make it a QTexteStream
	QTextStream css_stream(&css_file);
	// We get all the lines in one big QString
	QString css_string(css_stream.readAll());
	// Now it's time to close the QFile
	css_file.close();
	
	PVLOG_HEAVYDEBUG("The current CSS for the GUI is the following :\n%s\n-----------------------------\n", qPrintable(css_string));
 	
	// Now we can set the StyleSheet of the application.
	//app.setStyleSheet(css_string);
	

	pv_main_window.show();
	
	pv_main_window.setStyleSheet(css_string);

	if (vm.count("project")) {
		QString prj_path = QString::fromLocal8Bit(vm["project"].as<std::string>().c_str());
		pv_main_window.load_project(prj_path);
	}
	else 
	if (files.size() > 0) {
		pv_main_window.load_files(files, format);
	}

	int ret = app.exec();

	return ret;
}
//! [0]
