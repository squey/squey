#define QT_WIDGETS_LIB 1 // very important for Qt to instanciate a QApplication
#include <QTest>

#include <QFile>
#include <QKeyEvent>
#include <qtest_widgets.h>

#include <import_export.h>

#include <pvkernel/filter/PVPluginsLoad.h>
#include <pvkernel/rush/PVPluginsLoad.h>
#include <squey/common.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <PVMainWindow.h>
#include <pvguiqt/common.h>
#include <pvguiqt/PVExportSelectionDlg.h>

ImportExportTest::ImportExportTest()
{
    setenv("QTWEBENGINE_CHROMIUM_FLAGS", "--no-sandbox", 1); // see  https://bugs.chromium.org/p/chromium/issues/detail?id=638180
    setenv("PVKERNEL_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libpvkernel/plugins", 0);
    setenv("SQUEY_PLUGIN_PATH", SQUEY_BUILD_DIRECTORY "/libsquey/plugins", 0);

    Squey::common::load_filters();
    PVGuiQt::common::register_displays();
}

void ImportExportTest::import_file()
{
    App::PVMainWindow main_window;
    main_window.show();
    main_window.raise();

    bool success = false;
    QString tmp_export_filename(std::tmpnam(nullptr));

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

    QTimer::singleShot(0, [&]() {
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

void ImportExportTest::import_pcap()
{
    App::PVMainWindow main_window;
    main_window.show();
    main_window.raise();

    bool success = false;
    QString tmp_export_filename(std::tmpnam(nullptr));

    QPushButton* pcap_btn = main_window.findChild<QPushButton*>("pcap");

    connect(&main_window, &App::PVMainWindow::change_of_current_view_Signal, [&](){

        QTimer::singleShot(0, [&]() {
            PVGuiQt::PVExportSelectionDlg* export_file_dialog = main_window.findChild<PVGuiQt::PVExportSelectionDlg*>();

            connect(export_file_dialog, &PVGuiQt::PVExportSelectionDlg::selection_exported, [&]() 
            {
                QByteArray sha256sum;
                QFile f(tmp_export_filename+ ".pcap");
                if (f.open(QIODevice::ReadOnly)) {
                    QCryptographicHash hash(QCryptographicHash::Sha256);
                    if (hash.addData(&f)) {
                        sha256sum = hash.result();
                    }
                }
                f.remove();

                QCOMPARE(sha256sum.toHex(), QString("25a72bdf10339f2c29916920c8b9501d294923108de8f29b19aba7cc001ab60d"));
                success = true;
            });

            QLineEdit * lineEdit = qobject_cast<QLineEdit*>(export_file_dialog->focusWidget());
            lineEdit->setText(tmp_export_filename);

            QCheckBox* open_wireshark = export_file_dialog->findChild<QCheckBox*>("open_pcap_checkbox");
            open_wireshark->setChecked(false);

            QKeyEvent press(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
            qApp->sendEvent(export_file_dialog, &press);
        });
        main_window.export_selection_Slot();
    });

    QTimer::singleShot(0, [&]() {
        QPushButton* add_btn = main_window.findChild<QPushButton*>("add_button");

        QTimer::singleShot(0, [&]() {
            PVWidgets::PVFileDialog* import_pcap_dialog = main_window.findChildren<PVWidgets::PVFileDialog*>().at(1);
            QLineEdit * lineEdit = qobject_cast<QLineEdit*>(import_pcap_dialog->findChild<QLineEdit*>());
            QString source_path = QString(SQUEY_SOURCE_DIRECTORY) + "/tests/files/sources/http.pcap";
            lineEdit->setText(source_path);

            QKeyEvent press(QEvent::KeyPress, Qt::Key_Return, Qt::NoModifier);
            qApp->sendEvent(import_pcap_dialog, &press);

            QTimer::singleShot(0, [&]() {
                QPushButton* process_btn = main_window.findChild<QPushButton*>("process_import_button");
                QTest::mouseClick(process_btn, Qt::LeftButton, Qt::NoModifier);
             });
        });

        QTest::mouseClick(add_btn, Qt::LeftButton, Qt::NoModifier);
    });

    QTest::mouseClick(pcap_btn, Qt::LeftButton, Qt::NoModifier);

    main_window.disconnect();

    QCOMPARE(success, true);

}

QTEST_MAIN(ImportExportTest)
