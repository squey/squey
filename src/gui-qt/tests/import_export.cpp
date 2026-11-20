#define QT_WIDGETS_LIB 1 // very important for Qt to instanciate a QApplication
#include <QTest>

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <functional>
#include <algorithm>

#include <QDateTime>
#include <QFile>
#include <QFileDialog>
#include <QInputDialog>
#include <QLocale>
#include <QMessageBox>
#include <QRegularExpression>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QTimeZone>
#include <QTimer>
#include <QDropEvent>
#include <QMimeData>
#include <QUrl>
#include <qtest_widgets.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>
#include <vector>

#ifdef __linux__
#include <csignal>
#include <sys/resource.h>
#endif

#include <import_export.h>

#include <pvbase/general.h>
#include <pvkernel/core/PVStreamingCompressor.h>
#include <pvkernel/core/PVKernelSettings.h>
#include <pvkernel/core/PVSettings.h>
#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVInputType.h>
#include <pvkernel/rush/PVNrawCacheManager.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <squey/common.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvguiqt/PVAnalysisBreadcrumb.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <squey/PVAnalysisHistory.h>
#include <squey/PVView.h>
#include <QToolButton>
#include <pvkernel/widgets/PVImportWidget.h>
#include <pvkernel/rush/PVSourceDescription.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <pvguiqt/PVImportPage.h>
#include <pvkernel/core/PVTaskProgress.h>
#include <QAbstractButton>
#include <QPointer>
#include <QPushButton>
#include <QRadioButton>
#include <QTabBar>
#include <QMenuBar>
#include <QLabel>
#include <PVImportController.h>
#include <PVMainWindow.h>
#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/common.h>
#include <pvguiqt/PVExportSelectionDlg.h>

// Drives a modal dialog from *outside* its own exec() loop.
//
// A modal exec() blocks the code that opened it, so a test has to act from
// within that nested event loop. QTimer::singleShot(0) posted beforehand is not
// enough: the single event can fire before the dialog is up, or be consumed
// while exec() is still installing its loop, and the dialog then stays open
// until the test times out.
//
// Polling covers that, but through *two* channels rather than one: a repeating
// QTimer, and a worker thread reposting the same attempt with
// QMetaObject::invokeMethod(..., Qt::QueuedConnection). Both were observed
// carrying the test alone where the other stalled -- posted events kept flowing
// under the offscreen platform on Windows while timer events did not, and the
// export dialog is usually driven by the posted channel on Linux. Whichever
// arrives first wins; the other stops at the _driven flag.
//
// 'attempt' runs on the GUI thread and returns true once it has driven the
// dialog to completion. It must be idempotent: it is called repeatedly,
// including before the dialog exists, and returns false until it can act.
class ModalDriver {
  public:
	ModalDriver(QObject* context, std::function<bool()> attempt)
	    : _context(context), _attempt(std::move(attempt))
	{
	}

	void start()
	{
		_timer.setInterval(10);
		QObject::connect(&_timer, &QTimer::timeout, _context, [this]() { run(); });
		_timer.start();

		_worker = std::thread([this]() {
			while (not _stop.load() and not _driven.load()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
				// A driver is destroyed as soon as the dialog it drives has closed,
				// which can leave one of these calls still queued. Hand the lambda a
				// weak handle so a late one does nothing instead of reviving a dead
				// object: it is dropped in the destructor, and both run on the GUI
				// thread, so they cannot interleave.
				QMetaObject::invokeMethod(
				    _context,
				    [this, alive = std::weak_ptr<int>(_alive)]() {
					    if (not alive.expired()) {
						    run();
					    }
				    },
				    Qt::QueuedConnection);
			}
		});
	}

	~ModalDriver()
	{
		_stop.store(true);
		if (_worker.joinable()) {
			_worker.join();
		}
		_alive.reset();
	}

  private:
	void run()
	{
		if (_driven.load()) {
			return;
		}
		if (_attempt()) {
			_driven.store(true);
			_timer.stop();
		}
	}

	QObject* _context;
	std::function<bool()> _attempt;
	QTimer _timer;
	std::thread _worker;
	std::atomic<bool> _stop{false};
	std::atomic<bool> _driven{false};
	std::shared_ptr<int> _alive = std::make_shared<int>(0);
};

namespace
{

// Imports run in the import page of the main window, once the call that asked for them returned.
bool wait_for_import(App::PVMainWindow& main_window)
{
	return QTest::qWaitFor([&]() { return not main_window.import_controller().busy(); }, 120000);
}

// Waits for the import, with no dialog meanwhile.
bool wait_for_import_without_dialog(App::PVMainWindow& main_window)
{
	bool modal = false;
	const bool imported = QTest::qWaitFor(
	    [&]() {
		    modal |= qApp->activeModalWidget() != nullptr;
		    return not main_window.import_controller().busy();
	    },
	    120000);
	return imported and not modal;
}

// Opens the files as a command line does, and imports them as the import page asks.
void import_files(App::PVMainWindow& main_window, QStringList const& files)
{
	main_window.load_files(files);
	main_window._projects_tab_widget->import_page().findChild<QPushButton*>("import_import")->click();
}

// What the messages of the import page say.
QString import_messages(App::PVMainWindow& main_window)
{
	QStringList texts;
	for (const QFrame* message : main_window.findChildren<QFrame*>("import_message")) {
		for (const QLabel* label : message->findChildren<QLabel*>()) {
			texts << label->text();
		}
	}
	return texts.join('\n');
}

} // namespace

ImportExportTest::ImportExportTest()
{
    PVCore::setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox", 1); // see  https://bugs.chromium.org/p/chromium/issues/detail?id=638180
    PVCore::setenv("PVKERNEL_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
    PVCore::setenv("SQUEY_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libsquey/plugins", 0);
    // TESTS_SOURCE_DIR, not SQUEY_SOURCE_DIRECTORY: the latter is the absolute
    // source path of the build sandbox, which does not exist on the machine
    // running a cross-compiled test.
    PVCore::setenv("SQUEY_PCAP_PROFILES_PATH", TESTS_SOURCE_DIR "/libpvkernel/plugins/common/pcap/profiles", 0);

    Squey::common::load_filters();
    PVGuiQt::common::register_displays();

}

void ImportExportTest::import_file()
{
    App::PVMainWindow main_window;
    main_window.show();
    main_window.raise();

    bool success = false;
    QTemporaryDir tmp_dir;
    QVERIFY(tmp_dir.isValid());
    QString tmp_export_filename = tmp_dir.filePath("export");

    const QString source_path = QString(TEST_FOLDER) + "/picviz/enum_mapping.csv";
    QVERIFY2(QFileInfo::exists(source_path), qPrintable(source_path));

    // The export dialog is opened by export_selection_Slot() once the imported
    // view becomes current. Just trigger it here; the driver below closes it.
    connect(&main_window, &App::PVMainWindow::change_of_current_view_Signal, [&]() {
        main_window.export_selection_Slot();
    });

    bool export_connected = false;

    ModalDriver export_driver(&main_window, [&]() -> bool {
        // qApp->activeModalWidget() is the reliable handle for a modal dialog:
        // main_window.findChild() never located the import dialog on Windows CI.
        auto* dlg = qobject_cast<PVGuiQt::PVExportSelectionDlg*>(qApp->activeModalWidget());
        if (dlg == nullptr) {
            return false;
        }
        if (not export_connected) {
            connect(dlg, &PVGuiQt::PVExportSelectionDlg::selection_exported, [&]() {
                // Hash the exported rows, not the archive holding them: the
                // compression is delegated to whichever tool the platform ships
                // (see PVStreamingCompressor), so the bytes of the .gz change
                // when that tool is updated while the export itself does not.
                const QString exported_path = tmp_export_filename + ".csv.gz";
                QByteArray sha256sum;
                try {
                    PVCore::PVStreamingDecompressor decompressor(exported_path.toStdString());
                    QCryptographicHash hash(QCryptographicHash::Sha256);
                    char buffer[64 * 1024];
                    while (true) {
                        auto [uncompressed, compressed] =
                            decompressor.read(buffer, sizeof(buffer));
                        if (uncompressed == 0) {
                            break;
                        }
                        hash.addData(QByteArrayView(buffer, uncompressed));
                    }
                    sha256sum = hash.result();
                } catch (const PVCore::PVStreamingDecompressorError& e) {
                    QFAIL(e.what());
                }
                QFile::remove(exported_path);

                QCOMPARE(sha256sum.toHex(), QString("7bf9a8257c004c57b3a4c707bc965181c983439de757f792183e1c206f21fef8"));
                success = true;
            });
            export_connected = true;
        }

        // Locate the filename field explicitly rather than via focusWidget(),
        // which is null under the offscreen platform on macOS/Windows.
        QLineEdit* lineEdit = dlg->findChild<QLineEdit*>("fileNameEdit");
        if (lineEdit == nullptr) {
            return false;
        }
        lineEdit->setText(tmp_export_filename);

        // The default name filter is platform-dependent (".csv.gz" on Linux but
        // ".csv.zip" on macOS/Windows, see PVExportSelectionDlg), while the
        // checksum above expects the gz export: select the gz filter explicitly.
        QComboBox* filter_combo = dlg->findChild<QComboBox*>("fileTypeCombo");
        if (filter_combo == nullptr) {
            return false;
        }
        const QString gz_filter(".csv.gz files (*.csv.gz)");
        const int filter_index = filter_combo->findText(gz_filter);
        if (filter_index < 0) {
            return false;
        }
        filter_combo->setCurrentIndex(filter_index);
        Q_EMIT filter_combo->textActivated(gz_filter);

        // QFileDialog redeclares accept() as protected, hence the QDialog upcast.
        static_cast<QDialog*>(dlg)->accept();
        return not dlg->isVisible();
    });

    export_driver.start();

    // load_files() is the command-line import path (see main.cpp): it opens the import page
    // with the file, whose format sits next to it. The export dialog is still driven for real,
    // through the modal export flow it triggers.
    import_files(main_window, {source_path});
    QVERIFY(wait_for_import(main_window));

    QCOMPARE(success, true);
}

// What the history is worth only shows once it is wired to the window: a step
// has to be recorded by the action the user triggers, the breadcrumb has to
// hear about it, and going back has to put the rows back on screen.
void ImportExportTest::undo_redo()
{
    // The application stylesheet, because widgets measure differently under it
    // than under the plain style, and it is those measurements the breadcrumb
    // row is laid out from: a row that only comes out right under one of the
    // two would look right here and wrong in the application.
    //
    // Read straight from the resources rather than asked of PVTheme, whose
    // setter writes the chosen scheme into the user's configuration -- which a
    // test has no business doing, and which several tests at once have no
    // business doing to the same file. And set here rather than for the whole
    // class: the import tests drive modal dialogs, and dressing those up made
    // them crash under a loaded machine.
    QFile stylesheet(":/theme-light.qss");
    if (stylesheet.open(QFile::ReadOnly)) {
        qApp->setStyleSheet(QString::fromUtf8(stylesheet.readAll()));
    }

    // The locale the application runs under, see main.cpp, whatever the one of
    // the machine: the C locale groups no digits.
    struct CLocale {
        QLocale previous;
        CLocale() { QLocale::setDefault(QLocale::c()); }
        ~CLocale() { QLocale::setDefault(previous); }
    } c_locale;

    App::PVMainWindow main_window;
    main_window.show();
    main_window.raise();

    const QString source_path = QString(TEST_FOLDER) + "/picviz/enum_mapping.csv";
    QVERIFY2(QFileInfo::exists(source_path), qPrintable(source_path));
    import_files(main_window, {source_path});
    QVERIFY(wait_for_import(main_window));

    Squey::PVView* view = main_window.current_view();
    QVERIFY(view != nullptr);

    Squey::PVAnalysisHistory& history = main_window.get_root().history();
    QCOMPARE(history.size(), size_t(0));

    auto* breadcrumb = main_window.findChild<PVGuiQt::PVAnalysisBreadcrumb*>();
    QVERIFY(breadcrumb != nullptr);

    const size_t all_rows = view->get_real_output_selection().bit_count();
    QVERIFY(all_rows > 0);

    main_window.resize(1100, 700);
    QTest::qWait(200);
    const int empty_row = breadcrumb->height();
    QVERIFY(empty_row > 0);

    main_window.selection_none_Slot();

    // The step the action opened, plus the state it started from.
    QCOMPARE(history.size(), size_t(2));
    QCOMPARE(history.position(), size_t(1));
    QCOMPARE(view->get_real_output_selection().bit_count(), size_t(0));

    // The crumb carries the name the action gave it, which is how one can tell
    // the breadcrumb followed rather than merely existing. It shows an icon and
    // no text, so the name is in the tooltip -- as rich text, the three things a
    // crumb says being ruled off from one another.
    QStringList crumbs;
    for (QToolButton* button : breadcrumb->findChildren<QToolButton*>()) {
        if (not button->toolTip().isEmpty()) {
            crumbs << button->toolTip();
        }
    }
    QVERIFY2(std::any_of(crumbs.begin(), crumbs.end(),
                         [](QString const& t) { return t.contains("Empty selection"); }),
             qPrintable(crumbs.join(" | ")));

    // And what it left selected, digits grouped and as a share of the rows.
    const QString all_selected = QString(">%1 event(s) selected (100.0%)<")
                                     .arg(QLocale(QLocale::English).toString(qulonglong(all_rows)));
    QVERIFY2(std::any_of(crumbs.begin(), crumbs.end(),
                         [&](QString const& t) { return t.contains(all_selected); }),
             qPrintable(crumbs.join(" | ")));
    QVERIFY2(std::any_of(crumbs.begin(), crumbs.end(),
                         [](QString const& t) { return t.contains(">0 event(s) selected (0.0%)<"); }),
             qPrintable(crumbs.join(" | ")));

    // The strip is meant to cost as little height as the toolbar row above it,
    // and to cost the same before and after: a row that settles only once a
    // crumb has joined it jumps under the user on their first step. Waited for,
    // because a height read before the layout has run is the height from before.
    QTest::qWait(200);
    QVERIFY2(breadcrumb->height() <= 24, qPrintable(QString::number(breadcrumb->height())));
    QCOMPARE(breadcrumb->height(), empty_row);

    main_window.undo_Slot();
    QCOMPARE(history.position(), size_t(0));
    QCOMPARE(view->get_real_output_selection().bit_count(), all_rows);

    main_window.redo_Slot();
    QCOMPARE(history.position(), size_t(1));
    QCOMPARE(view->get_real_output_selection().bit_count(), size_t(0));

    // Acting after having gone back drops what lay ahead.
    main_window.undo_Slot();
    main_window.selection_all_Slot();
    QCOMPARE(history.size(), size_t(2));
    QCOMPARE(view->get_real_output_selection().bit_count(), all_rows);
}

// Several files of the same name are told apart by the names the user gives
// their sources, which a double click on the tab of a source asks for.
void ImportExportTest::rename_source()
{
	App::PVMainWindow main_window;
	main_window.show();
	main_window.raise();

	const QString source_path = QString(TEST_FOLDER) + "/picviz/enum_mapping.csv";
	QVERIFY2(QFileInfo::exists(source_path), qPrintable(source_path));
	import_files(main_window, {source_path});
	QVERIFY(wait_for_import(main_window));

	Squey::PVView* view = main_window.current_view();
	QVERIFY(view != nullptr);
	Squey::PVSource& source = view->get_parent<Squey::PVSource>();
	QCOMPARE(source.get_name(), std::string("enum_mapping.csv"));

	auto* tab_bar = main_window.findChild<PVGuiQt::PVSceneTabBar*>();
	QVERIFY(tab_bar != nullptr);
	QCOMPARE(tab_bar->tabText(0), QString("enum_mapping.csv"));

	// Longer than a tab shows, which then cuts it short and gives it whole in its tooltip.
	const QString name = "Enumerations mapped from the reference";
	ModalDriver rename_driver(&main_window, [&]() -> bool {
		auto* dlg = qobject_cast<QInputDialog*>(qApp->activeModalWidget());
		if (dlg == nullptr) {
			return false;
		}
		if (dlg->textValue() != QString("enum_mapping.csv")) {
			dlg->reject();
			return true;
		}
		dlg->setTextValue(name);
		dlg->accept();
		return not dlg->isVisible();
	});
	rename_driver.start();

	QTest::mouseDClick(tab_bar, Qt::LeftButton, Qt::NoModifier, tab_bar->tabRect(0).center());

	QCOMPARE(source.get_name(), name.toStdString());
	QCOMPARE(tab_bar->tabText(0), name.left(30) + "...");
	QCOMPARE(tab_bar->tabToolTip(0), name);
}

void ImportExportTest::import_pcap()
{
    App::PVMainWindow main_window;
    main_window.show();
    main_window.raise();

    bool pcap_exported = false;
    bool csv_exported = false;
    QTemporaryDir tmp_dir;
    QVERIFY(tmp_dir.isValid());
    QString tmp_export_filename = tmp_dir.filePath("export");

    // Drives one round through the export dialog with the given name filter, and
    // hands the exported file over to 'check'.
    auto run_export = [&](const QString& name_filter, const std::function<void()>& check) {
        bool connected = false;
        ModalDriver export_driver(&main_window, [&]() -> bool {
            auto* dlg = qobject_cast<PVGuiQt::PVExportSelectionDlg*>(qApp->activeModalWidget());
            if (dlg == nullptr) {
                return false;
            }
            if (not connected) {
                connect(dlg, &PVGuiQt::PVExportSelectionDlg::selection_exported, check);
                connected = true;
            }
            QLineEdit* lineEdit = dlg->findChild<QLineEdit*>("fileNameEdit");
            if (lineEdit == nullptr) {
                return false;
            }

            if (QCheckBox* open_wireshark = dlg->findChild<QCheckBox*>("open_pcap_checkbox")) {
                open_wireshark->setChecked(false);
            }

            // Picking the format takes simulating a real selection: neither
            // selectNameFilter() nor setCurrentIndex() emits filterSelected, the signal
            // the dialog listens to in order to swap the source exporter for the CSV
            // one -- the former is a plain setter, and QFileDialog wires its combo box
            // to textActivated(), which only a user's pick triggers. Setting the index
            // keeps the widget consistent, emitting textActivated() does the work.
            QComboBox* filter_combo = dlg->findChild<QComboBox*>("fileTypeCombo");
            if (filter_combo == nullptr) {
                return false;
            }
            const int filter_index = filter_combo->findText(name_filter);
            if (filter_index < 0) {
                return false;
            }
            filter_combo->setCurrentIndex(filter_index);
            // A real pick emits both: QFileDialog turns textActivated() into
            // filterSelected(), which is what swaps the exporter and the default
            // suffix, but it applies the filter itself from activated(). Emitting
            // only the first left the second export round writing a .pcap again on
            // Windows, and the .csv the check looks for was never produced.
            Q_EMIT filter_combo->activated(filter_index);
            Q_EMIT filter_combo->textActivated(name_filter);

            // Name last, as a user would: applying the filter re-navigates the
            // dialog, and a name typed before that loses its directory. That is how
            // the second round ended up writing to the home directory on Windows
            // while the check waited for the file in the temporary one.
            lineEdit->setText(tmp_export_filename);
            qInfo() << "export round" << name_filter << "-> default suffix"
                    << dlg->defaultSuffix() << ", selected" << dlg->selectedFiles();

            static_cast<QDialog*>(dlg)->accept();
            return not dlg->isVisible();
        });
        export_driver.start();
        main_window.export_selection_Slot();
    };

    connect(&main_window, &App::PVMainWindow::change_of_current_view_Signal, [&](){

        // Native pcap export: rewrites the captured packets.
        run_export(".pcap files (*.pcap)", [&]() {
            QByteArray sha256sum;
            QFile f(tmp_export_filename + ".pcap");
            if (f.open(QIODevice::ReadOnly)) {
                QCryptographicHash hash(QCryptographicHash::Sha256);
                if (hash.addData(&f)) {
                    sha256sum = hash.result();
                }
            }
            f.remove();

            QCOMPARE(sha256sum.toHex(), QString("25a72bdf10339f2c29916920c8b9501d294923108de8f29b19aba7cc001ab60d"));
            pcap_exported = true;
        });

        // CSV export: the pcap exporter above rewrites packets straight from the
        // capture and never reads the datetime_us column, so only this round puts
        // that column under test.
        run_export(".csv files (*.csv)", [&]() {
            QFile f(tmp_export_filename + ".csv");
            // Say what the export actually produced when it is not there: the file
            // going missing on Windows says nothing on its own, and whether the
            // dialog wrote it under another name or the export gave up half way
            // are two very different bugs.
            if (not f.exists()) {
                QStringList found = QDir(tmp_dir.path()).entryList(QDir::Files);
                QFAIL(qPrintable(QString("expected '%1' but the export directory holds: %2")
                                     .arg(f.fileName(),
                                          found.isEmpty() ? QString("(nothing)") : found.join(", "))));
            }
            QVERIFY(f.open(QIODevice::ReadOnly));
            const QString csv = QString::fromUtf8(f.readAll());
            f.remove();

            // frame.time must come back as a real temporal column. tshark hands it
            // over as ISO 8601 in UTC with nanoseconds ("...311224000+0000"); a
            // datetime_us that parses it reformats to microseconds ("...311224+0000").
            // If the profile format did not match the capture -- the bug this guards
            // against -- every value would stay invalid and the raw nanosecond string
            // would pass straight through untouched. Six fractional digits are the
            // proof the column was parsed rather than kept verbatim.
            const QRegularExpression microsecond_utc(
                R"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}\+0000)");
            const QRegularExpression nanosecond_utc(R"(\.\d{9}\+0000)");
            QVERIFY(microsecond_utc.match(csv).hasMatch());
            QVERIFY(not nanosecond_utc.match(csv).hasMatch());
            QVERIFY(not csv.contains("bad value"));
            csv_exported = true;
        });
    });

    const QString source_path = QString(TEST_FOLDER) + "/sources/http.pcap";
    QVERIFY2(QFileInfo::exists(source_path), qPrintable(source_path));

    // load_files() hands the capture to the pcap import dialog, in the import page, which
    // reads the fields of the capture before it can import.
    main_window.load_files({source_path});
    PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
    QVERIFY(page.content() != nullptr);
    QVERIFY(page.findChild<QPushButton*>("process_import_button")->isHidden());
    QPushButton* import = page.findChild<QPushButton*>("import_import");
    QVERIFY(QTest::qWaitFor([&]() { return import->isEnabled(); }, 60000));
    import->click();
    QVERIFY(wait_for_import(main_window));

    main_window.disconnect();

    QVERIFY(pcap_exported);
    QVERIFY(csv_exported);
}

namespace
{

void write_csv(QString const& path, int columns)
{
	QFile file(path);
	QVERIFY(file.open(QIODevice::WriteOnly));
	for (int row = 0; row < 100; row++) {
		QStringList fields;
		for (int col = 0; col < columns; col++) {
			fields << QString::number(row * columns + col);
		}
		file.write((fields.join(',') + '\n').toUtf8());
	}
}

// Imports the files through the import page, which detects the formats of those without one next
// to them, as one source each or one of all: no dialog opens meanwhile.
void import_detecting_formats(App::PVMainWindow& main_window,
                              QStringList const& files,
                              bool source_per_file)
{
	main_window.load_files(files);
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(page.content() != nullptr);
	page.findChild<QRadioButton*>("import_source_per_file")->setChecked(source_per_file);
	QPushButton* import = page.findChild<QPushButton*>("import_import");
	QVERIFY(import->isEnabled());
	import->click();
	QVERIFY(wait_for_import_without_dialog(main_window));
}
} // namespace

void ImportExportTest::import_guessed_formats()
{
	App::PVMainWindow main_window;
	main_window.show();
	main_window.raise();

	// Two files without a format, of different widths: neither format reads the
	// other file.
	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QStringList files = {tmp_dir.filePath("narrow.csv"), tmp_dir.filePath("wide.csv")};
	for (int i = 0; i < files.size(); i++) {
		write_csv(files[i], 3 + 2 * i);
	}
	if (QTest::currentTestFailed()) {
		return;
	}

	import_detecting_formats(main_window, files, true);
	if (QTest::currentTestFailed()) {
		return;
	}

	// A source of its own for each file, read with the format detected from it, which is not
	// written next to the data.
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(files.size()));
	for (Squey::PVSource const* source : sources) {
		QCOMPARE(source->get_inputs().size(), qsizetype(1));
		const QString path = source->get_inputs().front()->human_name();
		QCOMPARE(source->get_original_format().get_axes().size(),
		         path.endsWith("narrow.csv") ? qsizetype(3) : qsizetype(5));
		QVERIFY(source->get_original_format().get_full_path().isEmpty());
		QVERIFY(not QFile::exists(path + ".format"));
	}
}

void ImportExportTest::import_guessed_formats_without_nraw_dir()
{
	if (qEnvironmentVariableIsEmpty("SQUEY_NRAW_DIR")) {
		QSKIP("SQUEY_NRAW_DIR is not set: this test would replace the machine's collections");
	}
	// A file in place of the directory of the collections: no directory can be
	// created under it, whoever runs the test.
	const QString nraw_dir = PVRush::PVNrawCacheManager::nraw_dir();
	QVERIFY(QDir(nraw_dir).removeRecursively());
	QFile in_place(nraw_dir);
	QVERIFY(in_place.open(QIODevice::WriteOnly));
	in_place.close();
	// For the tests that run after this one.
	const auto restore = qScopeGuard([&nraw_dir]() {
		QFile::remove(nraw_dir);
		QDir().mkpath(nraw_dir);
	});

	App::PVMainWindow main_window;
	main_window.show();
	main_window.raise();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString file = tmp_dir.filePath("guessed.csv");
	write_csv(file, 3);
	if (QTest::currentTestFailed()) {
		return;
	}

	// The detection of the format needs no collection, the import cannot create it: it fails,
	// and says where.
	import_detecting_formats(main_window, {file}, false);
	if (QTest::currentTestFailed()) {
		return;
	}

	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains("Unable to create a temporary directory in " +
	                           QDir::toNativeSeparators(nraw_dir)),
	         qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());
}

// An import that fails part way says why, whichever stage failed: here the NRaw, whose files
// cannot grow, as on a full disk.
void ImportExportTest::import_failure_reported()
{
#ifndef __linux__
	QSKIP("Relies on what fallocate() does past RLIMIT_FSIZE on Linux");
#else
	App::PVMainWindow main_window;
	main_window.show();

	{
		rlimit previous_limit;
		QVERIFY(getrlimit(RLIMIT_FSIZE, &previous_limit) == 0);
		// Past the limit, the kernel refuses to grow a file once it has sent SIGXFSZ, which
		// would end the test.
		const auto previous_handler = std::signal(SIGXFSZ, SIG_IGN);
		const auto restore = qScopeGuard([&]() {
			setrlimit(RLIMIT_FSIZE, &previous_limit);
			std::signal(SIGXFSZ, previous_handler);
		});
		rlimit limit = previous_limit;
		limit.rlim_cur = 64 * 1024;
		QVERIFY(setrlimit(RLIMIT_FSIZE, &limit) == 0);

		import_files(main_window, {QString(TEST_FOLDER) + "/picviz/enum_mapping.csv"});
		QVERIFY(wait_for_import(main_window));
	}

	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains(PVRush::PVNrawCacheManager::nraw_dir()), qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());
#endif
}

// The axes of the dates, decimals and words that the import page detects are mapped as their
// types call for.
void ImportExportTest::import_guessed_types()
{
	App::PVMainWindow main_window;
	main_window.show();
	main_window.raise();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString path = tmp_dir.filePath("readings.log");
	QFile file(path);
	QVERIFY(file.open(QIODevice::WriteOnly));
	const char* const places[] = {"north", "south", "east", "west"};
	const QDateTime start(QDate(2022, 12, 8), QTime(13, 3, 3),
	                      QTimeZone::fromSecondsAheadOfUtc(3600));
	for (int row = 0; row < 100; row++) {
		file.write(start.addSecs(60 * row).toString(Qt::ISODate).toUtf8() + ';' +
		           QByteArray::number(9.812 + row * 0.125) + ';' + places[row % 4] + '\n');
	}
	file.close();

	main_window.import_type(LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file"));
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	page.content()->add_paths({path});
	QPushButton* import = page.findChild<QPushButton*>("import_import");
	QVERIFY(import->isEnabled());
	import->click();
	QVERIFY(wait_for_import_without_dialog(main_window));

	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	const QList<PVRush::PVAxisFormat>& axes = sources.front()->get_format().get_axes();
	QCOMPARE(axes.size(), qsizetype(3));
	QCOMPARE(axes[0].get_type(), QString("time"));
	QCOMPARE(axes[0].get_mapping(), QString("default"));
	QCOMPARE(axes[1].get_type(), QString("number_double"));
	QCOMPARE(axes[1].get_mapping(), QString("default"));
	QCOMPARE(axes[2].get_type(), QString("string"));
	QCOMPARE(axes[2].get_mapping(), QString("string"));
}

namespace
{

constexpr const char* ROWS_FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field><axis name="a" type="number_uint32"/></field>
  <field><axis name="b" type="number_uint32"/></field>
  <field><axis name="c" type="number_uint32"/></field>
 </splitter>
</param>
)";

// @p row_count rows of three numbers, with the format next to them, which the import finds.
void write_rows(QString const& path, int row_count)
{
	QFile file(path);
	QVERIFY(file.open(QIODevice::WriteOnly));
	QByteArray rows;
	for (int row = 0; row < row_count; row++) {
		rows += QByteArray::number(row) + ',' + QByteArray::number(row + 1) + ',' +
		        QByteArray::number(row + 2) + '\n';
		if (rows.size() > (1 << 20)) {
			file.write(rows);
			rows.clear();
		}
	}
	file.write(rows);
	QFile format(path + ".format");
	QVERIFY(format.open(QIODevice::WriteOnly));
	format.write(ROWS_FORMAT);
}

// The button @p text of the progress of the import page.
QPushButton* progress_button(App::PVMainWindow& main_window, QString const& text)
{
	auto* progress =
	    main_window._projects_tab_widget->import_page().findChild<QWidget*>("import_progress");
	for (QPushButton* button : progress->findChildren<QPushButton*>()) {
		if (button->text().trimmed() == text) {
			return button;
		}
	}
	return nullptr;
}

// The button @p text of the message of the import page that asks a question.
QPushButton* message_button(App::PVMainWindow& main_window, QString const& text)
{
	for (QFrame* message : main_window.findChildren<QFrame*>("import_message")) {
		for (QPushButton* button : message->findChildren<QPushButton*>()) {
			if (button->text().trimmed() == text) {
				return button;
			}
		}
	}
	return nullptr;
}

} // namespace

// The extraction and the processing show in the import page of the main window, which asks
// nothing, while the rest of the window waits.
void ImportExportTest::import_in_page()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	write_rows(csv, 1000);
	if (QTest::currentTestFailed()) {
		return;
	}

	import_files(main_window, {csv});
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QPushButton* back = page.findChild<QPushButton*>("import_back");
	QVERIFY(main_window.import_controller().busy());
	QVERIFY(page.isVisible());
	QVERIFY(not main_window.menuBar()->isEnabled());
	QVERIFY(not back->isEnabled());
	bool modal = false;
	QVERIFY(QTest::qWaitFor(
	    [&]() {
		    modal |= qApp->activeModalWidget() != nullptr;
		    return not main_window.import_controller().busy();
	    },
	    120000));
	QVERIFY(not modal);

	QVERIFY(not page.isVisible());
	QVERIFY(main_window.menuBar()->isEnabled());
	QVERIFY(back->isEnabled());
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QCOMPARE(sources.front()->get_row_count(), PVRow(1000));
}

// Files dropped on the main window open in the import page, as a command line opens them, but
// not over what the page asks for.
void ImportExportTest::import_dropped_files()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	write_rows(csv, 100);
	if (QTest::currentTestFailed()) {
		return;
	}

	QMimeData mime;
	mime.setUrls({QUrl::fromLocalFile(csv)});
	const auto drop = [&]() {
		QDragEnterEvent enter(QPointF(10, 10), Qt::CopyAction, &mime, Qt::LeftButton,
		                      Qt::NoModifier);
		QCoreApplication::sendEvent(&main_window, &enter);
		if (not enter.isAccepted()) {
			return false;
		}
		QDropEvent dropped(QPointF(10, 10), Qt::CopyAction, &mime, Qt::LeftButton, Qt::NoModifier);
		QCoreApplication::sendEvent(&main_window, &dropped);
		return dropped.isAccepted();
	};
	QVERIFY(drop());
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(
	    QTest::qWaitFor([&]() { return page.content() != nullptr and page.isVisible(); }, 10000));
	QCOMPARE(page.content()->sources().size(), size_t(1));
	QCOMPARE(page.content()->sources().front().first.size(), qsizetype(1));
	QVERIFY(not drop());

	page.findChild<QPushButton*>("import_import")->click();
	QVERIFY(wait_for_import_without_dialog(main_window));
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QCOMPARE(sources.front()->get_row_count(), PVRow(100));
}

// What cannot be imported is told in the import page, which stays until the user leaves it.
void ImportExportTest::import_error_in_page()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("empty.csv");
	write_rows(csv, 0);
	if (QTest::currentTestFailed()) {
		return;
	}

	import_files(main_window, {csv});
	QVERIFY(wait_for_import(main_window));
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(page.isVisible());
	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains("were empty"), qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());

	// With what it asked for the import, to try again.
	QVERIFY(page.content() != nullptr and page.content()->isVisible());
	QPushButton* cancel = page.findChild<QPushButton*>("import_cancel");
	QVERIFY(cancel->isVisible());
	cancel->click();
	QVERIFY(not page.isVisible());
	QVERIFY(import_messages(main_window).isEmpty());
}

// A format whose types do not allow its mappings is told in the import page.
void ImportExportTest::import_incompatible_format()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	write_rows(csv, 100);
	if (QTest::currentTestFailed()) {
		return;
	}
	// The mapping of the hours of times, for numbers.
	QFile format(csv + ".format");
	QVERIFY(format.open(QIODevice::WriteOnly | QIODevice::Truncate));
	format.write(
	    QByteArray(ROWS_FORMAT)
	        .replace(R"(<axis name="a" type="number_uint32"/>)",
	                 R"(<axis name="a" type="number_uint32"><mapping mode="24h"/></axis>)"));
	format.close();

	import_files(main_window, {csv});
	QVERIFY(wait_for_import_without_dialog(main_window));
	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains("not compatible"), qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());
}

// Discarding an import asks first, in the page: going on resumes it, discarding drops the source.
// The window cannot close meanwhile.
void ImportExportTest::discard_import()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	write_rows(csv, 100000);
	if (QTest::currentTestFailed()) {
		return;
	}

	import_files(main_window, {csv});
	QVERIFY(not main_window.close());
	QPushButton* discard = progress_button(main_window, "Discard");
	QVERIFY(discard != nullptr and discard->isVisible());
	discard->click();
	QPushButton* resume = message_button(main_window, "Continue");
	QVERIFY(resume != nullptr);
	QVERIFY(not discard->isEnabled());
	resume->click();
	QVERIFY(discard->isEnabled());
	discard->click();
	QPushButton* confirm = message_button(main_window, "Discard");
	QVERIFY(confirm != nullptr);
	confirm->click();

	QVERIFY(wait_for_import(main_window));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());
	// Only what told the window could not close.
	QVERIFY2(not import_messages(main_window).contains("Discard"),
	         qPrintable(import_messages(main_window)));
}

// Stopping an import keeps the events extracted so far, which are processed.
void ImportExportTest::stop_import()
{
	// Small chunks, two at once: the import goes a few chunks further than where it is paused or
	// stopped.
	QVERIFY(PVCore::settings::import_chunk_size.set_value(65536));
	QVERIFY(PVCore::settings::import_parallel_chunks.set_value(2));
	const auto restore = qScopeGuard([]() {
		PVCore::PVSettings::get().reset(PVCore::settings::import_chunk_size.key());
		PVCore::PVSettings::get().reset(PVCore::settings::import_parallel_chunks.key());
	});

	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	constexpr int row_count = 400000;
	write_rows(csv, row_count);
	if (QTest::currentTestFailed()) {
		return;
	}

	// The question of the discard pauses the import before it can go far, whatever the speed of
	// the machine, and lets the chunks it holds go through meanwhile.
	import_files(main_window, {csv});
	progress_button(main_window, "Discard")->click();
	QTest::qWait(200);
	message_button(main_window, "Continue")->click();
	progress_button(main_window, "Stop and process")->click();

	QVERIFY(wait_for_import(main_window));
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QVERIFY(sources.front()->get_row_count() > 0);
	QVERIFY2(sources.front()->get_row_count() < PVRow(row_count),
	         qPrintable(QString::number(sources.front()->get_row_count())));
}

namespace
{

void write_json(QString const& path, QByteArray const& lines)
{
	QFile file(path);
	QVERIFY(file.open(QIODevice::WriteOnly));
	file.write(lines);
}


} // namespace

// JSON files given together, as from the command line, open in the import page, which reads them
// for their columns, and imports them as one source once asked to.
void ImportExportTest::import_json_in_page()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString first = tmp_dir.filePath("first.json");
	const QString second = tmp_dir.filePath("second.json");
	write_json(first, "{\"x\": 1, \"s\": \"a\"}\n");
	write_json(second, "{\"x\": 2}\n{\"x\": 3, \"s\": \"c\"}\n");
	if (QTest::currentTestFailed()) {
		return;
	}

	main_window.load_files({first, second});
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(page.isVisible());
	QVERIFY(page.content() != nullptr);
	bool modal = false;
	QVERIFY(QTest::qWaitFor(
	    [&]() {
		    modal |= qApp->activeModalWidget() != nullptr;
		    return page.content()->is_ready();
	    },
	    60000));
	QVERIFY(not modal);
	page.findChild<QPushButton*>("import_import")->click();
	QVERIFY(wait_for_import_without_dialog(main_window));

	QVERIFY(not page.isVisible());
	QVERIFY(page.content() == nullptr);
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	const PVRush::PVNraw& nraw = sources.front()->get_rushnraw();
	QCOMPARE(nraw.row_count(), PVRow(3));
	QCOMPARE(nraw.at_string(2, PVCol(0)), std::string("second.json"));
}

// A source of the recent items, of JSON files, has no schema: it is read again out of the GUI
// thread, while the page shows the progress.
void ImportExportTest::import_recent_json()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString path = tmp_dir.filePath("recent.json");
	write_json(path, "{\"x\": 1, \"s\": \"a\"}\n{\"x\": 2, \"s\": \"b\"}\n");
	if (QTest::currentTestFailed()) {
		return;
	}

	PVRush::PVInputType_p json = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json");
	main_window.load_source_from_description_Slot(PVRush::PVSourceDescription(
	    {json->load_input_from_string({path.toStdString()}, false)},
	    PVRush::PVSourceCreatorFactory::get_by_input_type(json), PVRush::PVFormat()));
	QVERIFY(wait_for_import_without_dialog(main_window));

	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QCOMPARE(sources.front()->get_rushnraw().row_count(), PVRow(2));
}

// The parameters of an import, as D-Bus gives them, are read out of the GUI thread, and what is
// wrong with them is told in the page.
void ImportExportTest::import_params()
{
	App::PVMainWindow main_window;
	main_window.show();

	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString path = tmp_dir.filePath("params.json");
	write_json(path, "{\"x\": 1, \"s\": \"a\"}\n{\"x\": 2, \"s\": \"b\"}\n{\"x\": 3, \"s\": \"c\"}\n");
	if (QTest::currentTestFailed()) {
		return;
	}

	PVRush::PVInputType_p json = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json");
	const auto params = [&path](QString const& row_filter) {
		return QJsonDocument(QJsonObject{{"paths", QJsonArray{path}}, {"row_filter", row_filter}})
		    .toJson(QJsonDocument::Compact);
	};
	main_window.import_controller().import_params(json, params("x > 1"));
	QVERIFY(wait_for_import_without_dialog(main_window));
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QCOMPARE(sources.front()->get_rushnraw().row_count(), PVRow(2));

	main_window.import_controller().import_params(json, params("nope > 1"));
	QVERIFY(wait_for_import_without_dialog(main_window));
	QCOMPARE(main_window.get_root().get_children<Squey::PVSource>().size(), size_t(1));
	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains("nope"), qPrintable(messages));
}

// The Back button of the import page leaves it in its tab, with what it shows, for the page shown
// before; the close button of its tab drops what it shows.
void ImportExportTest::leave_import_page()
{
	App::PVMainWindow main_window;
	main_window.show();
	PVGuiQt::PVProjectsTabWidget& projects = *main_window._projects_tab_widget;
	PVGuiQt::PVImportPage& page = projects.import_page();
	QTabBar* tabs = projects.findChild<QTabBar*>();
	constexpr int import_tab = PVGuiQt::PVProjectsTabWidget::IMPORT_PAGE_INDEX;
	projects.project_new();
	const int collection_tab = tabs->currentIndex();
	QCOMPARE(collection_tab, PVGuiQt::PVProjectsTabWidget::FIRST_PROJECT_INDEX);

	main_window.import_type(LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json"));
	const QPointer<PVWidgets::PVImportWidget> content = page.content();
	QVERIFY(content != nullptr);
	QVERIFY(page.isVisible());
	page.findChild<QPushButton*>("import_back")->click();
	QCOMPARE(tabs->currentIndex(), collection_tab);
	QVERIFY(tabs->isTabVisible(import_tab));
	QCOMPARE(page.content(), content.data());

	tabs->setCurrentIndex(import_tab);
	QVERIFY(page.isVisible());
	QWidget* close = tabs->tabButton(import_tab, QTabBar::LeftSide);
	if (close == nullptr) {
		close = tabs->tabButton(import_tab, QTabBar::RightSide);
	}
	QVERIFY(qobject_cast<QAbstractButton*>(close) != nullptr);
	qobject_cast<QAbstractButton*>(close)->click();
	QCOMPARE(tabs->currentIndex(), collection_tab);
	QVERIFY(not tabs->isTabVisible(import_tab));
	QVERIFY(QTest::qWaitFor([&]() { return content.isNull(); }, 10000));
}

namespace
{

// What the import page shows, which reads its inputs until it is canceled.
class ReadingContent : public PVWidgets::PVImportWidget
{
  public:
	bool is_ready() const override { return false; }
	std::vector<source_t> sources() const override { return {}; }

	void read()
	{
		run_task(
		    "Reading",
		    [](PVCore::PVTaskProgress& progress) {
			    while (not progress.canceled()) {
				    std::this_thread::sleep_for(std::chrono::milliseconds(10));
			    }
		    },
		    [](bool) {});
	}
};

} // namespace

// Closing the import page while what it shows reads its inputs stops the reading first.
void ImportExportTest::close_import_page_while_reading()
{
	App::PVMainWindow main_window;
	main_window.show();
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();

	const QPointer<ReadingContent> content = new ReadingContent;
	QVERIFY(main_window.import_controller().edit(
	    LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("json"), content));
	content->read();
	QVERIFY(page.task_running());
	page.findChild<QPushButton*>("import_cancel")->click();
	QVERIFY(page.isVisible());
	QCOMPARE(page.content(), content.data());

	QVERIFY(QTest::qWaitFor([&]() { return not page.isVisible(); }, 10000));
	QVERIFY(QTest::qWaitFor([&]() { return content.isNull(); }, 10000));
	QVERIFY(not main_window.import_controller().busy());
}

// The window going away while an import runs, as the application quits, stops the import before
// it drops the source.
void ImportExportTest::destroy_window_while_importing()
{
	QTemporaryDir tmp_dir;
	QVERIFY(tmp_dir.isValid());
	const QString csv = tmp_dir.filePath("rows.csv");
	write_rows(csv, 400000);
	if (QTest::currentTestFailed()) {
		return;
	}

	auto main_window = std::make_unique<App::PVMainWindow>();
	main_window->show();
	import_files(*main_window, {csv});
	// For the extraction to be under way.
	QTest::qWait(10);
	QVERIFY(main_window->import_controller().busy());
	main_window.reset();
}

QTEST_MAIN(ImportExportTest)
