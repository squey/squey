/**
 * \file PVAxesCombinationWidget.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H
#define PVINSPECTOR_PVAXESCOMBINATIONWIDGET_H

#include <QDialog>
#include <QComboBox>
#include <QList>

#include <picviz/PVAxesCombination.h>
#include <picviz/PVView_types.h>

#include <pvguiqt/ui_PVAxesCombinationWidget.h>
#include <pvguiqt/ui_PVAxesCombinationWidgetSelRange.h>

namespace PVGuiQt {

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
	PVAxesCombinationWidget(Picviz::PVAxesCombination& axes_combination, Picviz::PVView* view = NULL, QWidget* parent = 0);

public:
	void save_current_combination();
	void restore_saved_combination();

	void reset_used_axes();

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
	void set_selection_from_cols(QList<PVCol> const& cols);

protected slots:
	void axis_add_Slot();
	void axis_up_Slot();
	void axis_down_Slot();
	void axis_move_Slot();
	void axis_remove_Slot();
	void reset_comb_Slot();
	void sel_singleton_Slot();
	void sel_range_Slot();
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

class PVAxesCombinationWidgetSelRange: public QDialog, Ui::PVAxesCombinationWidgetSelRange
{
	Q_OBJECT
public:
	enum values_source_t
	{
		mapped = 0,
		plotted
	};

public:
	PVAxesCombinationWidgetSelRange(QWidget* parent = NULL);

public:
	bool get_range(float& min, float& max);
	bool reversed();
	double rate();
	values_source_t get_source();
};

}

#endif
