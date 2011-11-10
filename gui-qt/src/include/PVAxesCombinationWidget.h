#ifndef PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H
#define PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H

#include <QDialog>
#include <QComboBox>
#include <QList>

#include <picviz/PVAxesCombination.h>
#include "../ui_PVAxesCombinationWidget.h"
#include <picviz/PVView_types.h>

namespace PVInspector {

class PVAxesCombinationWidget: public QWidget, Ui::PVAxesCombinationWidget
{
	Q_OBJECT

private:
	class PVMoveToDlg: public QDialog
	{
	public:
		PVMoveToDlg(PVAxesCombinationWidget* parent);

	public:
		PVCol get_dest_col(PVCol org);
		void update_axes();

	private:
		QComboBox* _after_combo;
		QComboBox* _axes_combo;
		PVAxesCombinationWidget* _parent;
	};

public:
	PVAxesCombinationWidget(Picviz::PVAxesCombination& axes_combination, QWidget* parent = 0, Picviz::PVView* view = NULL);

public:
	void save_current_combination();
	void restore_saved_combination();
public slots:
	void update_orig_axes();
	void update_used_axes();
	void update_all();

protected:
	PVCol get_original_axis_selected();
	QString get_original_axis_selected_name();
	QVector<PVCol> get_used_axes_selected();
	bool is_used_axis_selected();
	bool is_original_axis_selected();
	static QVector<PVCol> get_list_selection(QListWidget* widget);

protected slots:
	void axis_add_Slot();
	void axis_up_Slot();
	void axis_down_Slot();
	void axis_move_Slot();
	void axis_remove_Slot();
	void reset_comb_Slot();
	void sel_singleton_Slot();
	void sort_Slot();

signals:
	void axes_combination_changed();
	void axes_count_changed();

protected:
	Picviz::PVAxesCombination& _axes_combination;
	Picviz::PVAxesCombination _saved_combination;
	PVMoveToDlg* _move_dlg;
	Picviz::PVView* _view;
};

}

#endif
