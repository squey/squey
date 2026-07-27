#define QT_WIDGETS_LIB 1 // very important for Qt to instanciate a QApplication
#include <QTest>

#include <QComboBox>
#include <QDir>
#include <QFileInfo>
#include <functional>
#include <QFile>
#include <QFileDialog>
#include <QRegularExpression>
#include <QTemporaryDir>
#include <QTimer>
#include <qtest_widgets.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include <import_export.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <squey/common.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <PVMainWindow.h>
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
                QByteArray sha256sum;
                QFile f(tmp_export_filename + ".csv.gz");
                if (f.open(QFile::ReadOnly)) {
                    QCryptographicHash hash(QCryptographicHash::Sha256);
                    if (hash.addData(&f)) {
                        sha256sum = hash.result();
                    }
                }
                f.remove();

                QCOMPARE(sha256sum.toHex(), QString("5e3122873a465857982c76f7f70ef4728f9680c5931a9bb0afbfa582359dc5d6"));
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

    QCOMPARE(success, true);
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

    main_window.disconnect();

    QVERIFY(pcap_exported);
    QVERIFY(csv_exported);
}

QTEST_MAIN(ImportExportTest)
