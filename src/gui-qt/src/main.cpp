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

#include <pvbase/general.h>
#include "PVSingleInstanceApplication.h"

//! [0]
#include <QFile>
#include <QString>
#include <QObject>
#include <QWidget>
#include <QSplashScreen>
#include <QShortcut>
#include <QMessageBox>

#include <PVMainWindow.h>

#include <string>
#include <vector>
#include <iostream>
#ifndef _WIN32
#include <sys/resource.h>
#else
#include <io.h> // TODO: CHECK IF NEEDED
#include <fcntl.h> // TODO: CHECK IF NEEDED
#include <stdio.h> // TODO: CHECK IF NEEDED
#endif

#include <pvkernel/core/PVConfig.h>
#include <pvkernel/core/squey_intrin.h>
#ifdef __linux__
#include <pvkernel/core/segfault_handler.h>
#endif
#include <pvkernel/core/qobject_helpers.h>
#include <pvkernel/opencl/common.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/core/PVUtils.h>

#include <squey/common.h>

#include <pvparallelview/PVParallelView.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVNrawDirectoryMessageBox.h>
#include <pvguiqt/PVChangelogMessage.h>

#include <pvdisplays/PVDisplaysImpl.h>

#include <boost/program_options.hpp>
#include <boost/dll/runtime_symbol_info.hpp>

#include <filesystem>
#ifdef _WIN32
#include <pvkernel/core/PVUtilitiesDecompressor.h>
#endif


//#include <QtWebEngineWidgets/QWebEngineView>

// #ifdef USE_UNIKEY
// #include <UniKeyFR.h>
// #endif
class DisplaysFocusInEventFilter : public QObject
{
  protected:
	bool eventFilter(QObject* obj, QEvent* event) override
	{
		if (event->type() == QEvent::FocusIn) {
			// Is the widget a PVViewDisplay?
			auto* display = qobject_cast<PVGuiQt::PVViewDisplay*>(obj);
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
	bool eventFilter(QObject* watched, QEvent* event) override
	{
		if (event->type() == QEvent::Move) {
			QWidget* window = qobject_cast<QWidget*>(watched);
			if (window &&
			    QLatin1String("QShapedPixmapWidget") == window->metaObject()->className()) {
				window->setAttribute(Qt::WA_TranslucentBackground);
				window->clearMask();
			}
		}
		return false;
	}
};

namespace bpo = boost::program_options;

int run_squey(App::PVSingleInstanceApplication& app, int argc, char* argv[])
{
	// Program options
	bpo::options_description desc_opts("Options");
	desc_opts.add_options()("help", "produce help message")(
	    "format", bpo::value<std::string>(),
	    "path to the format to use. Default is automatic discovery.");
	bpo::options_description hidden_opts("Hidden options");
	hidden_opts.add_options()("input-file", bpo::value<std::vector<std::string>>(),
	                          "path to the file to load");
	hidden_opts.add_options()(
	    "product", bpo::value<std::string>()->default_value("squey"), "product name");
	bpo::options_description all_opts;
	all_opts.add(desc_opts).add(hidden_opts);

	bpo::positional_options_description p;
	p.add("input-file", -1);

	bpo::variables_map vm;
	bpo::store(bpo::command_line_parser(argc, argv).options(all_opts).positional(p).run(), vm);
	bpo::notify(vm);

	if (vm.count("help")) {
		std::cerr << "Squey " << SQUEY_CURRENT_VERSION_STR << std::endl << std::endl;
		std::cerr << "Usage: " << argv[0] << " [--format format] [file [file...]]" << std::endl;
		std::cerr << desc_opts << std::endl;
		return 1;
	}

	QString format;
	if (vm.count("format")) {
		std::string format_arg = vm["format"].as<std::string>();
		format = QString::fromLocal8Bit(format_arg.c_str(), format_arg.size());
	}

	QStringList files;
	if (vm.count("input-file")) {
		std::vector<std::string> files_arg = vm["input-file"].as<std::vector<std::string>>();
		files.reserve(files_arg.size());
		// Convert file path to unicode
		for (std::string const& arg : files_arg) {
			files.push_back(QString::fromLocal8Bit(arg.c_str(), arg.size()));
		}
	}

	std::string product_name;
	product_name = vm["product"].as<std::string>();
	PVCore::PVConfig::set_product_name(product_name);

	// Init theme
	PVCore::PVTheme::init();

	// Check at least two CPU cores are available
	if (std::thread::hardware_concurrency() == 1) {

		QMessageBox::critical(
		    nullptr, QObject::tr("Not enough available CPU cores"),
		    QObject::tr("At least 2 CPU cores are required in order to run the software.<br><br>"
		                "If you are running the software inside a virtual machine or a container, "
		                "please increase the number of CPU cores."));
		return 1;
	}

	// Ensure nraw tmp directory exists and is writable
	{
		PVNrawDirectoryMessageBox nraw_tmp_checker;
	}

	QSplashScreen splash(QPixmap(":/splash-screen"));

	auto* vl = new QVBoxLayout(&splash);

	auto* task_label = new QLabel();
	task_label->setStyleSheet("background: transparent;");
	task_label->setAlignment(Qt::AlignLeft | Qt::AlignTop);

	auto* version_label = new QLabel(QString("Squey ") + SQUEY_CURRENT_VERSION_STR);
	version_label->setStyleSheet("background: transparent;");
	version_label->setAlignment(Qt::AlignRight | Qt::AlignBottom);

	vl->addWidget(task_label);
	vl->addSpacing(0);
	vl->addWidget(version_label);

	splash.show();
	app.processEvents();

#ifdef _WIN32
	task_label->setText(QObject::tr("Uncompressing utilities..."));
	splash.repaint();
	app.processEvents();
	PVCore::PVUtilitiesDecompressor();
#endif

	task_label->setText(QObject::tr("Initializing backends..."));
	splash.repaint();
	app.processEvents();
	PVParallelView::common::RAII_backend_init backend_resources;

	task_label->setText(QObject::tr("Loading plugins..."));
	splash.repaint();
	app.processEvents();
	Squey::common::load_filters();
	PVGuiQt::common::register_displays();

	task_label->setText(QObject::tr("Cleaning temporary files..."));
	splash.repaint();
	app.processEvents();
	PVRush::PVNrawCacheManager::get().delete_unused_cache();

	task_label->setText(QObject::tr("Finishing initialization..."));
	splash.repaint();
	app.processEvents();

	app.setWindowIcon(QIcon(":/squey"));
	app.installEventFilter(new DragNDropTransparencyHack());
	app.installEventFilter(new DisplaysFocusInEventFilter());

	App::PVMainWindow pv_mw;
	pv_mw.setAttribute(Qt::WA_TranslucentBackground);
	pv_mw.showMaximized();
	splash.finish(&pv_mw);

	// Show changelog if software version has changed
	{
		PVGuiQt::PVChangelogMessage changelog_msg(&pv_mw);
	}

	pv_mw.set_window_title_with_filename();

	QObject::connect(
	    &app,
		&App::PVSingleInstanceApplication::files_opened,
		&pv_mw,
		std::bind(&App::PVMainWindow::load_files, &pv_mw, std::placeholders::_1, "")
	);
	if (files.size() > 0) {
		pv_mw.load_files(files, format);
	}

#if 1 // Taking screenshots is not supported under Wayland
	/* set the screenshot shortcuts as global shortcuts
	 */
	QShortcut* sc;
	sc = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, &QShortcut::activated, &pv_mw,
	                 &App::PVMainWindow::get_screenshot_widget);

	sc = new QShortcut(QKeySequence(Qt::SHIFT | Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, &QShortcut::activated, &pv_mw,
	                 &App::PVMainWindow::get_screenshot_window);

	sc = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, &QShortcut::activated, &pv_mw,
	                 &App::PVMainWindow::get_screenshot_desktop);
#endif

	return app.exec();
}

int main(int argc, char* argv[])
{
	App::PVSingleInstanceApplication app(argc, argv);
	if (app.is_running()) {
	    return 0;
	}

#ifdef __APPLE__
	QCoreApplication::setAttribute(Qt::AA_DontShowIconsInMenus, false);
#endif

#ifdef _WIN32
	// Update PATH
	boost::filesystem::path exe_path = boost::dll::program_location();
	std::string libdir = exe_path.parent_path().string();
#ifdef SQUEY_DEVELOPER_MODE
	const char* path = PVCore::getenv("PATH");
	if (path) {
		libdir += ";" + std::string(path);
	}
#endif
	PVCore::setenv("PATH", libdir.c_str(), 1);

    // Attach console to parent process
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
    }
#endif

	setlocale(LC_ALL, "C.UTF-8");
	PVCore::setenv("OMP_TOOL", "disabled", 1); // Disable OMP_TOOL to avoid "Unable to find TSan function" errors
	PVCore::setenv("LANG", "C.UTF-8", 1);
	PVCore::setenv("TZ", "GMT", 1);
	tzset();

#ifdef __linux__
	init_segfault_handler();
#endif

	// Set the soft limit same as hard limit for number of possible files opened
#ifndef _WIN32
	rlimit ulimit_info;
	getrlimit(RLIMIT_NOFILE, &ulimit_info);
	ulimit_info.rlim_cur = ulimit_info.rlim_max;
	setrlimit(RLIMIT_NOFILE, &ulimit_info);
#endif

	return run_squey(app, argc, argv);
}
//! [0]
