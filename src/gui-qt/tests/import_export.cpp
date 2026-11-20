#define QT_WIDGETS_LIB 1 // very important for Qt to instanciate a QApplication
#include <QTest>

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <functional>
#include <algorithm>

#include <QFile>
#include <QFileDialog>
#include <QInputDialog>
#include <QLocale>
#include <QMessageBox>
#include <QRegularExpression>
#include <QScopeGuard>
#include <QTemporaryDir>
#include <QTimer>
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
#include <pvguiqt/PVImportPage.h>
#include <QPushButton>
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

    // Import programmatically, bypassing the file-selection dialog. That dialog
    // (PVImportFileDialog, a QFileDialog with an injected combo box) drives an
    // infinite size-hint propagation loop under the offscreen platform on
    // Windows CI -- an endless flood of "propagateSizeHints" warnings ending in a
    // stack overflow (0xc00000fd) -- and is not even reliably findable there.
    // load_files() is the command-line import path (see main.cpp); it reaches the
    // very same import_type() with no modal file dialog. The export dialog is
    // still driven for real, through the modal export flow it triggers.
    main_window.load_files({source_path});
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
    main_window.load_files({source_path});
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
	main_window.load_files({source_path});
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

    // load_files() feeds the capture straight into the pcap import wizard, which
    // opens *pre-populated* (PVInputTypePcap::create_widget_with_input_files) --
    // so there is no "add" button to click and, crucially, no nested modal
    // QFileDialog to drive (that QFileDialog is what stack-overflows the offscreen
    // platform on Windows, see import_file). The wizard is still modal: this
    // driver just clicks its "process" button once it is up.
    bool processed = false;
    ModalDriver wizard_driver(&main_window, [&]() -> bool {
        auto* modal = qApp->activeModalWidget();
        if (modal == nullptr) {
            return false;
        }
        QPushButton* process_btn = modal->findChild<QPushButton*>("process_import_button");
        if (process_btn == nullptr or not process_btn->isVisible()) {
            return false;
        }
        if (not processed) {
            processed = true;
            QTest::mouseClick(process_btn, Qt::LeftButton, Qt::NoModifier);
        }
        return true;
    });

    wizard_driver.start();

    main_window.load_files({source_path});
    QVERIFY(wait_for_import(main_window));

    main_window.disconnect();

    QVERIFY(pcap_exported);
    QVERIFY(csv_exported);
}

namespace
{
//! Answers to the questions of an import, each by a part of the text of its question.
using Answers = std::vector<std::pair<QString, QMessageBox::StandardButton>>;

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

// Imports the files as the import dialog hands them over with the default
// format. With no format next to the files, the import guesses one for each, and
// asks about it as a user would be asked: a question gets the first answer whose
// key is part of its text, and any other dialog fails the test.
void import_guessing_formats(App::PVMainWindow& main_window,
                             QStringList const& files,
                             Answers const& answers_by_question)
{
	PVRush::PVInputType_p in_file = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	PVRush::hash_formats formats;
	PVRush::PVInputType::list_inputs inputs;
	PVCore::PVArgumentList args;
	QString format;
	QVERIFY(in_file->create_widget_with_input_files(files, formats, inputs, format, args,
	                                                &main_window));
	QCOMPARE(format, QString(SQUEY_LOCAL_FORMAT_STR));

	QStringList unexpected;
	ModalDriver answers(&main_window, [&]() -> bool {
		QWidget* modal = qApp->activeModalWidget();
		if (modal == nullptr or not modal->isVisible()) {
			return false;
		}
		if (auto* box = qobject_cast<QMessageBox*>(modal)) {
			for (auto const& [question, answer] : answers_by_question) {
				if (box->text().contains(question)) {
					box->button(answer)->click();
					return false;
				}
			}
			unexpected << box->text() + " " + box->informativeText();
			box->reject();
		} else if (qobject_cast<QInputDialog*>(modal) or qobject_cast<QFileDialog*>(modal)) {
			unexpected << modal->windowTitle();
			static_cast<QDialog*>(modal)->reject();
		}
		// One question after the other, until the import is over.
		return false;
	});
	answers.start();

	PVRush::hash_format_creator format_creator;
	main_window.import_type(in_file, inputs, formats, format_creator, format);
	QVERIFY(wait_for_import(main_window));

	QVERIFY2(unexpected.isEmpty(), qPrintable(unexpected.join('\n')));
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

	// The format is only used once saved.
	import_guessing_formats(main_window, files,
	                        {{"Do you want to automatically add that splitter", QMessageBox::Yes},
	                         {"do you want to manually enter column names", QMessageBox::No},
	                         {"The format has been modified", QMessageBox::Save}});
	if (QTest::currentTestFailed()) {
		return;
	}

	// A source of its own for each file, read with the format guessed from it.
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(files.size()));
	for (Squey::PVSource const* source : sources) {
		QCOMPARE(source->get_inputs().size(), qsizetype(1));
		QCOMPARE(source->get_original_format().get_full_path(),
		         source->get_inputs().front()->human_name() + ".format");
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

	// The format builder previews the file as soon as it adds the guessed splitter,
	// from the event loop of the question that follows. Neither that preview nor the
	// import can create the collection: the import fails, and says where.
	import_guessing_formats(
	    main_window, {file},
	    {{"Do you want to automatically add that splitter", QMessageBox::Yes},
	     {"do you want to manually enter column names", QMessageBox::No},
	     {"The format has been modified", QMessageBox::Save},
	     {"Unable to create a temporary directory in " + QDir::toNativeSeparators(nraw_dir),
	      QMessageBox::Ok}});
	if (QTest::currentTestFailed()) {
		return;
	}

	// The preview asks in a dialog, the import tells in its page.
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

		main_window.load_files({QString(TEST_FOLDER) + "/picviz/enum_mapping.csv"});
		QVERIFY(wait_for_import(main_window));
	}

	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains(PVRush::PVNrawCacheManager::nraw_dir()), qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());
#endif
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
		if (button->text() == text) {
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
			if (button->text() == text) {
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

	main_window.load_files({csv});
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(main_window.import_controller().busy());
	QVERIFY(page.isVisible());
	QVERIFY(not main_window.menuBar()->isEnabled());
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
	const auto sources = main_window.get_root().get_children<Squey::PVSource>();
	QCOMPARE(sources.size(), size_t(1));
	QCOMPARE(sources.front()->get_row_count(), PVRow(1000));
}

// What cannot be imported is told in the import page, which stays until the user closes it.
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

	main_window.load_files({csv});
	QVERIFY(wait_for_import(main_window));
	PVGuiQt::PVImportPage& page = main_window._projects_tab_widget->import_page();
	QVERIFY(page.isVisible());
	const QString messages = import_messages(main_window);
	QVERIFY2(messages.contains("were empty"), qPrintable(messages));
	QVERIFY(main_window.get_root().get_children<Squey::PVSource>().empty());

	QPushButton* close = page.findChild<QPushButton*>("import_close");
	QVERIFY(close != nullptr and close->isVisible());
	close->click();
	QVERIFY(not page.isVisible());
	QVERIFY(import_messages(main_window).isEmpty());
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

	main_window.load_files({csv});
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
	main_window.load_files({csv});
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

QTEST_MAIN(ImportExportTest)
