/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
} // namespace PVWidgets

namespace Inendi
{
class PVAxesCombination;
class PVView;
class PVSelection;
} // namespace Inendi

namespace PVGuiQt
{

class PVAxesCombinationWidget;

/** Specific widget to export a selection to csv.
 *
 * Pop a QFileDialog to specify filename and the kind of export and do it.
 */
class PVExportSelectionDlg : public QFileDialog
{
	Q_OBJECT;

  public:
	/** Pop the FileDialog and perform the export.
	 *
	 * ExportSelectionDlg can't be created directly, creation and export are
	 * done in one step.
	 */
	static void export_selection(Inendi::PVView& view, const Inendi::PVSelection& sel);

  private: // Interfaces used to export the selection
	enum class AxisCombinationKind { ALL, CURRENT, CUSTOM };

	/** Create a FileDialog to export selection
	 *
	 * @param custom_axes_combination : Combination of axis selected using the view menu
	 * @param view : The view to export
	 * @param parent : parent widget (as usual in Qt)
	 */
	PVExportSelectionDlg(Inendi::PVAxesCombination& custom_axes_combination,
	                     Inendi::PVView& view,
	                     QWidget* parent = 0);

	/** Return the kind of axis combination we want to export. */
	AxisCombinationKind combination_kind() const;

	/** Separator to use between csv fields */
	QString separator_char() const { return _separator_char->keySequence().toString(); }

	/** Separator to use for quoted fields */
	QString quote_char() const { return _quote_char->keySequence().toString(); }

	/** Wether we want to export an header line or not. */
	inline bool export_columns_header() const { return _columns_header->isChecked(); }

  private Q_SLOTS:
	/** Enable or disable the button to edit custom axis exported. */
	void show_edit_axes_widget(bool show);

	/** Show the widget to edit axis combination */
	void show_axes_combination_widget();

  private:
	PVAxesCombinationWidget*
	    _axes_combination_widget; //!< The axis combination widget to select axis to export
	PVWidgets::QKeySequenceWidget* _quote_char;     //!< Character to use to quote a field
	PVWidgets::QKeySequenceWidget* _separator_char; //!< Character to use as a csv separator
	QCheckBox* _columns_header;          //!< Box to say if we want to export header line or not
	QCheckBox* _export_internal_values;  //!< Specify exporting plotted instead of nraw
	QPushButton* _edit_axes_combination; //!< The edit button to select custom axis
	QRadioButton* _all_axis;             //!< Buttom if all axis are exported
	QRadioButton* _current_axis;         //!< Button to export only axis from current view
	QRadioButton* _custom_axis;          //!< Button if custom selected axis are exported
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVEXPORTSELECTIONDLG_H__
