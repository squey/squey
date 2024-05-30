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

private:
    PVParallelView::common::RAII_backend_init backend_resources;
};

#endif // __IMPORT_EXPORT_H__
