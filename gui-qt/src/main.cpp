/**
 * \file main.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

//! [0]
#include <QApplication>
#include <QFile>
#include <QLocale>
#include <QResource>
#include <QString>
#include <QTextStream>
#include <QTextCodec>
#include <QObject>
#include <QWidget>
#include <QSplashScreen>
#include <QShortcut>

#include <PVMainWindow.h>
#include <PVCustomStyle.h>

#include <string>
#include <vector>
#include <iostream>
#include <sys/sysinfo.h>

#include <stdio.h>
//#include <dlfcn.h>

#include <time.h>
#include <pvkernel/core/picviz_intrin.h>
#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include <picviz/common.h>
#include <picviz/PVRoot.h>
#include <picviz/common.h>

#include <pvparallelview/PVParallelView.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVViewDisplay.h>

#include <pvdisplays/PVDisplaysImpl.h>

#include <boost/program_options.hpp>

#define JULY_5 1309856400

// #ifdef USE_UNIKEY
  // #include <UniKeyFR.h>
// #endif
class DisplaysFocusInEventFilter : public QObject
{
protected:
	bool eventFilter(QObject* obj, QEvent *event)
	{
		if (event->type() == QEvent::FocusIn) {
			// Is the widget a PVViewDisplay?
			PVGuiQt::PVViewDisplay* display = qobject_cast<PVGuiQt::PVViewDisplay*>(obj);
			if (!display) {
				// Or a child of a PVViewDisplay ?
				display = PVCore::get_qobject_parent_of_type<PVGuiQt::PVViewDisplay*>(obj);
			}
			if (display) {
				display->set_current_view();
			}
		}

		return QObject::eventFilter(obj, event);
	}

};

class DragNDropTransparencyHack : public QObject
{
public:
	bool eventFilter(QObject* watched, QEvent* event)
	{
		if (event->type() == QEvent::Move) {
			QWidget *window = qobject_cast<QWidget*>(watched);
			if (window && QLatin1String("QShapedPixmapWidget") == window->metaObject()->className()) {
				window->setAttribute(Qt::WA_TranslucentBackground);
				window->clearMask();
			}
		}
		return false;
	}
};

static __attribute__((noinline)) void __check__t()
{
	// License validity test : it's a simple "time" check
	if (time(NULL) >= CUSTOMER_RELEASE_EXPIRATION_DATE) {
		exit(0);
	}
}

static __attribute__((noinline)) void __check__m()
{
	struct sysinfo info;
	sysinfo(&info);

	if (info.totalram > (size_t) (CUSTOMER_CAPABILITY_MEMORY^4294967295) * 1 << (30)) {
		exit(0);
	}
}

namespace bpo = boost::program_options;

// #define NO_MAIN_WINDOW

int main(int argc, char *argv[])
{
	init_segfault_handler();
	PVCore::PVConfig::get().init_dirs();

#ifndef NO_MAIN_WINDOW
	QApplication app(argc, argv);
#endif
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

	__check__t();
	__check__m();

#ifndef NO_MAIN_WINDOW
	QSplashScreen splash(QPixmap(":/splash-screen"));

	QVBoxLayout *vl = new QVBoxLayout(&splash);

	QLabel *task_label = new QLabel();
	task_label->setAlignment(Qt::AlignLeft | Qt::AlignTop);

	QLabel *version_label = new QLabel(QString("Picviz Inspector ") + PICVIZ_CURRENT_VERSION_STR);
	version_label->setAlignment(Qt::AlignRight | Qt::AlignBottom);

	vl->addWidget(task_label);
	vl->addSpacing(0);
	vl->addWidget(version_label);

	splash.show();
	app.processEvents();
#endif

#ifdef CUDA
#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Initializing CUDA..."));
	splash.repaint();
	app.processEvents();
	PVParallelView::common::init_cuda();
#endif
#endif
#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Loading plugins..."));
	splash.repaint();
	app.processEvents();
#endif
	Picviz::common::load_filters();
#ifndef NO_MAIN_WINDOW
	PVGuiQt::common::register_displays();
#endif

#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Cleaning temporary files..."));
	splash.repaint();
	app.processEvents();
#endif
	PVRush::PVNrawCacheManager::get()->delete_unused_cache();

#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Finishing initialization..."));
	splash.repaint();
	app.processEvents();
#endif

	//app.setStyle(new PVInspector::PVCustomStyle());
#ifndef NO_MAIN_WINDOW
	PVInspector::PVMainWindow* pv_mw = new PVInspector::PVMainWindow();
#endif
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

#ifndef NO_MAIN_WINDOW
	app.setOrganizationName("PICVIZ Labs");
	app.setApplicationName("Picviz Inspector " PICVIZ_CURRENT_VERSION_STR);
	app.setWindowIcon(QIcon(":/picviz"));
	app.installEventFilter(new DragNDropTransparencyHack());
	app.installEventFilter(new DisplaysFocusInEventFilter());
#endif

#ifndef NO_MAIN_WINDOW
	pv_mw->show();
	splash.finish(pv_mw);

	if (vm.count("project")) {
		QString prj_path = QString::fromLocal8Bit(vm["project"].as<std::string>().c_str());
		pv_mw->load_project(prj_path);
	}
	else 
	if (files.size() > 0) {
		pv_mw->load_files(files, format);
	}
	else {
		// Set default title
		pv_mw->set_window_title_with_filename();
	}

	/* set the screenshot shortcuts as global shortcuts
	 */
	QShortcut* sc;
	sc = new QShortcut(QKeySequence(Qt::Key_P), pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()),
	                 pv_mw, SLOT(get_screenshot_widget()));

	sc = new QShortcut(QKeySequence(Qt::SHIFT + Qt::Key_P), pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()),
	                 pv_mw, SLOT(get_screenshot_window()));

	sc = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_P), pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()),
	                 pv_mw, SLOT(get_screenshot_desktop()));

	int ret = app.exec();
#else
	int ret = 0;
#endif

#ifndef NO_MAIN_WINDOW
	PVParallelView::common::release();
	PVDisplays::release();
#endif

	return ret;
}
//! [0]
