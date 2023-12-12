#define QT_WIDGETS_LIB 1 // very important for Qt to instanciate a QApplication
#include <QTest>

#include <QFile>
#include <QKeyEvent>
#include <qtest_widgets.h>

#include <PVMainWindow.h>
#include <pvparallelview/PVParallelView.h>

#include <import_export.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <squey/common.h>
#include <pvkernel/widgets/PVFileDialog.h>

#include <pvguiqt/common.h>
#include <pvguiqt/PVExportSelectionDlg.h>

void ImportExportTest::run_test()
{
    setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox", 1); // see  https://bugs.chromium.org/p/chromium/issues/detail?id=638180

    setenv("PVKERNEL_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
	setenv("SQUEY_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libsquey/plugins", 0);

    PVParallelView::common::RAII_backend_init backend_resources;
    Squey::common::load_filters();
    PVGuiQt::common::register_displays();

    bool success = false;
    QString tmp_export_filename(std::tmpnam(nullptr));

    App::PVMainWindow main_window = App::PVMainWindow();
    main_window.show();

    QPushButton* btn = main_window.findChild<QPushButton*>("file");

    connect(&main_window, &App::PVMainWindow::change_of_current_view_Signal, [&](){

        QTimer::singleShot(0, [&]() {
            PVGuiQt::PVExportSelectionDlg* export_file_dialog = main_window.findChild<PVGuiQt::PVExportSelectionDlg*>();

            connect(export_file_dialog, &PVGuiQt::PVExportSelectionDlg::selection_exported, [&]() 
            {
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

            QLineEdit * lineEdit = qobject_cast<QLineEdit*>(export_file_dialog->focusWidget());
            lineEdit->setText(tmp_export_filename);

            QKeyEvent press(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
            qApp->sendEvent(export_file_dialog, &press);
        });
        main_window.export_selection_Slot();
    });

    QTimer::singleShot(0, [&main_window]() {
        PVWidgets::PVFileDialog* import_file_dialog = main_window.findChild<PVWidgets::PVFileDialog*>("PVImportFileDialog");
        QLineEdit * lineEdit = qobject_cast<QLineEdit*>(import_file_dialog->focusWidget());
        QString source_path = QString(SQUEY_SOURCE_DIRECTORY) + "/tests/files/picviz/enum_mapping.csv";
        lineEdit->setText(source_path);

        QKeyEvent press(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
	    qApp->sendEvent(import_file_dialog, &press);
    });

    QTest::mouseClick(btn, Qt::LeftButton, Qt::NoModifier);

    QCOMPARE(success, true);
}

QTEST_MAIN(ImportExportTest)
