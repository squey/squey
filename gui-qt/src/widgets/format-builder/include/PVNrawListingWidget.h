#ifndef PVNRAWLISTINGWIDGET_H
#define PVNRAWLISTINGWIDGET_H

#include <pvcore/general.h>
#include <pvrush/PVInputType.h>

#include <QWidget>
#include <QLineEdit>
#include <QPushButton>
#include <QLabel>
#include <QTableView>

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
	void set_last_input(PVRush::PVInputType_p in_t = PVRush::PVInputType_p(), PVCore::PVArgument input = PVCore::PVArgument());
	void resize_columns_content();
	void unselect_column();
	void select_column(PVCol col);

protected:
	PVNrawListingModel* _nraw_model;
	QLineEdit* _ext_start;
	QLineEdit* _ext_end;
	QPushButton* _btn_preview;
	QLabel* _src_label; 
	QTableView* _nraw_table;
};

}

#endif

