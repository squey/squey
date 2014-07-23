/**
 * \file PVExportSelectionDlg.h
 *
 * Copyright (C) Picviz Labs 2014
 */

#ifndef __PVGUIQT_PVEXPORTSELECTIONDLG_H__
#define __PVGUIQT_PVEXPORTSELECTIONDLG_H__

#include <QFileDialog>
#include <QCheckBox>
#include <QRadioButton>

#include <pvkernel/widgets/qkeysequencewidget.h>

namespace PVWidgets
{
	class QKeySequenceWidget;
}

namespace Picviz
{
	class PVAxesCombination;
	class PVView;
	class PVSelection;
}

namespace PVGuiQt
{

class PVAxesCombinationWidget;

class PVExportSelectionDlg : public QFileDialog
{
	Q_OBJECT;

public:
	PVExportSelectionDlg(Picviz::PVAxesCombination& custom_axes_combination, Picviz::PVView& view, QWidget* parent = 0);

	static void export_selection(Picviz::PVView& view, const Picviz::PVSelection& sel);

private:
	inline bool export_columns_header() const { return _columns_header->checkState() == Qt::Checked; }
	inline bool use_custom_axes_combination() const { return !_use_current_axes_combination->isChecked(); }
	inline QString separator_char() const { return _separator_char->keySequence().toString(); }
	inline QString quote_char() const { return _quote_char->keySequence().toString(); }
	inline Picviz::PVAxesCombination& get_custom_axes_combination() const { return _custom_axes_combination; }

private slots:
	void show_axes_combination_widget(bool show);
	void edit_axes_combination();

private:
	Picviz::PVAxesCombination& _custom_axes_combination;
	PVWidgets::QKeySequenceWidget* _separator_char;
	PVWidgets::QKeySequenceWidget* _quote_char;
	QCheckBox* _columns_header;
	QRadioButton* _use_current_axes_combination;
	PVAxesCombinationWidget* _axes_combination_widget;
	QPushButton* _edit_axes_combination;
};

}

#endif // __PVGUIQT_PVEXPORTSELECTIONDLG_H__
