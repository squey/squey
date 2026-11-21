#ifndef __IMPORT_EXPORT_H__
#define __IMPORT_EXPORT_H__

#include <QObject>
#include <pvparallelview/PVParallelView.h>

class ImportExportTest : public QObject
{
    Q_OBJECT

public:
    ImportExportTest();

private Q_SLOTS:
    void import_file();
    void import_pcap();
    void import_guessed_formats();
    void undo_redo();
    void import_guessed_formats_without_nraw_dir();
    void import_failure_reported();
    void import_guessed_types();
    void rename_source();
    void import_in_page();
    void import_dropped_files();
    void import_error_in_page();
    void import_incompatible_format();
    void discard_import();
    void stop_import();
    void import_json_in_page();
    void import_recent_json();
    void import_params();
    void leave_import_page();
    void close_import_page_while_reading();
    void destroy_window_while_importing();
    void working_directory_out_of_reach();

private:
    PVParallelView::common::RAII_backend_init backend_resources;
};

#endif // __IMPORT_EXPORT_H__
