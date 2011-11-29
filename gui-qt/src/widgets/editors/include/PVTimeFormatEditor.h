#ifndef PVCORE_PVTIMEFORMATEDITOR_H
#define PVCORE_PVTIMEFORMATEDITOR_H

#include <QTextEdit>
#include <pvkernel/core/general.h>
#include <pvkernel/core/PVTimeFormatType.h>

namespace PVInspector {

class PVTimeFormatEditor: public QTextEdit
{
	Q_OBJECT
	Q_PROPERTY(PVCore::PVTimeFormatType time_formats READ get_time_formats WRITE set_time_formats USER true)

public:
	PVTimeFormatEditor(QWidget *parent = 0);

public:
	PVCore::PVTimeFormatType get_time_formats() const;
	void set_time_formats(PVCore::PVTimeFormatType const& tfs);
};

}

#endif
