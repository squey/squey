#ifndef PVNRAWLISTINGWIDGET_H
#define PVNRAWLISTINGWIDGET_H

#include <pvcore/general.h>
#include <QWidget>
#include <QLineEdit>
#include <QPushButton>

namespace PVInspector {

// Forward declaration
class PVNrawListingModel;

class PVNrawListingWidget: public QWidget
{
public:
	PVNrawListingWidget(PVNrawListingModel* nraw_model, QWidget* parent = NULL);

public:
	void connect_preview(QObject* receiver, const char* slot);
	void get_ext_args(PVRow& start, PVRow& end);

protected:
	PVNrawListingModel* _nraw_model;
	QLineEdit* _ext_start;
	QLineEdit* _ext_end;
	QPushButton* _btn_preview;
};

}

#endif

