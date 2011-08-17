#ifndef PVNRAWLISTINGWIDGET_H
#define PVNRAWLISTINGWIDGET_H

#include <pvkernel/core/general.h>
#include <pvkernel/rush/PVInputType.h>

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
	Q_OBJECT
public:
	PVNrawListingWidget(PVNrawListingModel* nraw_model, QWidget* parent = NULL);

public:
	void connect_preview(QObject* receiver, const char* slot);
	void connect_axes_name(QObject* receiver, const char* slot);
	void connect_axes_type(QObject* receiver, const char* slot);
	void get_ext_args(PVRow& start, PVRow& end);
	void set_last_input(PVRush::PVInputType_p in_t = PVRush::PVInputType_p(), PVCore::PVArgument input = PVCore::PVArgument());
	void resize_columns_content();
	void unselect_column();
	void select_column(PVCol col);

public slots:
	void nraw_custom_menu_Slot(const QPoint& pt);
	void set_axes_name_selected_row_Slot();
	void set_axes_type_selected_row_Slot();

signals:
	void set_axes_name_from_nraw(int row);
	void set_axes_type_from_nraw(int row);

protected:
	int get_selected_row();

protected:
	PVNrawListingModel* _nraw_model;
	QLineEdit* _ext_start;
	QLineEdit* _ext_end;
	QPushButton* _btn_preview;
	QLabel* _src_label; 
	QTableView* _nraw_table;
	QMenu* _ctxt_menu;
};

}

#endif

