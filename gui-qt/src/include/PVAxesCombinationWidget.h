#ifndef PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H
#define PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H

#include <picviz/PVAxesCombination.h>
#include "../ui_PVAxesCombinationWidget.h"

namespace PVInspector {

class PVAxesCombinationWidget: public QWidget, Ui::PVAxesCombinationWidget
{
	Q_OBJECT
public:
	PVAxesCombinationWidget(Picviz::PVAxesCombination& axes_combination, QWidget* parent = 0);

public:
	void save_current_combination();
	void restore_saved_combination();
	void update_used_axes();

protected:
	PVCol get_original_axis_selected();
	QString get_original_axis_selected_name();
	PVCol get_used_axis_selected();
	bool is_used_axis_selected();
	bool is_original_axis_selected();

protected slots:
	void axis_add_Slot();
	void axis_up_Slot();
	void axis_down_Slot();
	void axis_remove_Slot();

signals:
	void axes_combination_changed();
	void axes_count_changed();

protected:
	Picviz::PVAxesCombination& _axes_combination;
	Picviz::PVAxesCombination _saved_combination;
};

}

#endif
