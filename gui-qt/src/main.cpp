/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvbase/general.h>

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
#include <QMessageBox>

#include <PVMainWindow.h>

#include <string>
#include <vector>
#include <iostream>
#include <sys/resource.h>

#include <stdio.h>

#include <time.h>
#include <pvkernel/core/inendi_intrin.h>
#include <pvkernel/core/segfault_handler.h>
#include <pvkernel/core/qobject_helpers.h>
#include <License.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include <inendi/common.h>

#include <pvparallelview/PVParallelView.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVViewDisplay.h>

#include <pvdisplays/PVDisplaysImpl.h>

#include <boost/program_options.hpp>

static QString email_address = EMAIL_ADDRESS_CONTACT;

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

// #define NO_MAIN_WINDOW

int main(int argc, char* argv[])
{
	setlocale(LC_ALL, "C");
	setenv("LANG", "C", 1);
	setenv("TZ", "GMT", 1);
	tzset();

	init_segfault_handler();

	// Set the soft limit same as hard limit for number of possible files opened
	rlimit ulimit_info;
	getrlimit(RLIMIT_NOFILE, &ulimit_info);
	ulimit_info.rlim_cur = ulimit_info.rlim_max;
	setrlimit(RLIMIT_NOFILE, &ulimit_info);

	QString license_file = "/etc/inendi/licenses/inendi-inspector.lic";

#ifndef NO_MAIN_WINDOW
	QApplication app(argc, argv);

	if (not QFile(license_file).exists()) {
		QMessageBox::critical(nullptr, QObject::tr("INENDI-inspector"),
		                      QObject::tr("You don't have you license file : %1. If you have a "
		                                  "license file, rename "
		                                  "it with this name, otherwise contact : <a "
		                                  "href=\"mailto:%2?subject=%5BINENDI%5D\">%2</a>")
		                          .arg(license_file)
		                          .arg(email_address));
		return 1;
	}
#endif

	// Set location to check for license file.
	setenv("LM_LICENSE_FILE", license_file.toUtf8().constData(), 1);

	Inendi::Utils::License::RAII_InitLicense license_manager;

	Inendi::Utils::License::RAII_LicenseFeature full_program_license(INENDI_FLEX_PREFIX,
	                                                                 INENDI_FLEX_FEATURE);
	// Program options
	bpo::options_description desc_opts("Options");
	desc_opts.add_options()("help", "produce help message")(
	    "format", bpo::value<std::string>(),
	    "path to the format to use. Default is automatic discovery.");
	bpo::options_description hidden_opts("Hidden options");
	hidden_opts.add_options()("input-file", bpo::value<std::vector<std::string>>(),
	                          "path to the file to load");
	bpo::options_description all_opts;
	all_opts.add(desc_opts).add(hidden_opts);

	bpo::positional_options_description p;
	p.add("input-file", -1);

	bpo::variables_map vm;
	bpo::store(bpo::command_line_parser(argc, argv).options(all_opts).positional(p).run(), vm);
	bpo::notify(vm);

	if (vm.count("help")) {
		std::cerr << "INENDI Inspector " << INENDI_CURRENT_VERSION_STR << std::endl << std::endl;
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
		std::vector<std::string> files_arg = vm["input-file"].as<std::vector<std::string>>();
		files.reserve(files_arg.size());
		std::vector<std::string>::const_iterator it;
		// Convert file path to unicode
		for (it = files_arg.begin(); it != files_arg.end(); it++) {
			files.push_back(QString::fromLocal8Bit(it->c_str(), it->size()));
		}
	}

	Inendi::Utils::License::check_ram(INENDI_FLEX_PREFIX, INENDI_FLEX_FEATURE, INENDI_FLEX_MAXMEM);

#ifndef NO_MAIN_WINDOW
	QSplashScreen splash(QPixmap(":/splash-screen"));

	QVBoxLayout* vl = new QVBoxLayout(&splash);

	QLabel* task_label = new QLabel();
	task_label->setAlignment(Qt::AlignLeft | Qt::AlignTop);

	QLabel* version_label = new QLabel(QString("INENDI Inspector ") + INENDI_CURRENT_VERSION_STR);
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
	PVParallelView::common::RAII_cuda_init cuda_resources;
#endif
#endif
#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Loading plugins..."));
	splash.repaint();
	app.processEvents();
#endif
	Inendi::common::load_filters();
#ifndef NO_MAIN_WINDOW
	PVGuiQt::common::register_displays();
#endif

#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Cleaning temporary files..."));
	splash.repaint();
	app.processEvents();
#endif
	PVRush::PVNrawCacheManager::get().delete_unused_cache();

#ifndef NO_MAIN_WINDOW
	task_label->setText(QObject::tr("Finishing initialization..."));
	splash.repaint();
	app.processEvents();
#endif

	QString locale = QLocale::system().name();
	PVLOG_INFO("System locale: %s\n", qPrintable(locale));

	PVCore::PVIntrinsics::init_cpuid();
#ifdef __SSE4_1__
	PVLOG_INFO("Compiled with SSE 4.1 instructions\n");
	if (PVCore::PVIntrinsics::has_sse41()) {
		PVLOG_INFO("SSE4.1 is supported by this CPU.\n");
	} else {
		PVLOG_INFO("SSE4.1 is not supported by this CPU.\n");
	}
#endif

#ifndef NO_MAIN_WINDOW
	app.setOrganizationName("ESI Group");
	app.setApplicationName("INENDI Inspector " INENDI_CURRENT_VERSION_STR);
	app.setWindowIcon(QIcon(":/inendi"));
	app.installEventFilter(new DragNDropTransparencyHack());
	app.installEventFilter(new DisplaysFocusInEventFilter());
#endif

#ifndef NO_MAIN_WINDOW
	PVInspector::PVMainWindow pv_mw;
	pv_mw.show();
	splash.finish(&pv_mw);

	if (files.size() > 0) {
		pv_mw.load_files(files, format);
	} else {
		// Set default title
		pv_mw.set_window_title_with_filename();
	}

	/* set the screenshot shortcuts as global shortcuts
	 */
	QShortcut* sc;
	sc = new QShortcut(QKeySequence(Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()), &pv_mw, SLOT(get_screenshot_widget()));

	sc = new QShortcut(QKeySequence(Qt::SHIFT + Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()), &pv_mw, SLOT(get_screenshot_window()));

	sc = new QShortcut(QKeySequence(Qt::CTRL + Qt::Key_P), &pv_mw);
	sc->setContext(Qt::ApplicationShortcut);
	QObject::connect(sc, SIGNAL(activated()), &pv_mw, SLOT(get_screenshot_desktop()));

	return app.exec();
#else
	return 0;
#endif
}
//! [0]
