/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __PVGUIQT_PVEXPORTSELECTIONDLG_H__
#define __PVGUIQT_PVEXPORTSELECTIONDLG_H__

#include <QComboBox>
#include <QFileDialog>
#include <QCheckBox>
#include <QRadioButton>

#include <pvkernel/rush/PVExporter.h>

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

	/** Create a FileDialog to export selection
	 *
	 * @param custom_axes_combination : Combination of axis selected using the view menu
	 * @param view : The view to export
	 * @param parent : parent widget (as usual in Qt)
	 */
	PVExportSelectionDlg(Inendi::PVView& view, QWidget* parent = 0);

  public:
	PVRush::PVExporterBase& exporter()
	{
		return _is_source_exporter ? *_source_exporter : *_exporter;
	}

  private:
	bool _is_source_exporter;
	PVRush::PVExporterBase* _exporter = nullptr;
	PVRush::PVExporterBase* _source_exporter = nullptr; // exporter specific to the source, if any
};
} // namespace PVGuiQt

#endif // __PVGUIQT_PVEXPORTSELECTIONDLG_H__
