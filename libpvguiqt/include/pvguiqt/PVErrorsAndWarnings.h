#ifndef __PVGUIQT_PVERRORSANDWARNINGS_H__
#define __PVGUIQT_PVERRORSANDWARNINGS_H__

#include <QWidget>

namespace Squey {
class PVSource;
}

class PVErrorsAndWarnings : public QWidget
{
    Q_OBJECT

public:
    PVErrorsAndWarnings(Squey::PVSource* source, QWidget* invalid_events_dialog);

    static size_t invalid_columns_count(const Squey::PVSource* src);
};

#endif // __PVGUIQT_PVERRORSANDWARNINGS_H__