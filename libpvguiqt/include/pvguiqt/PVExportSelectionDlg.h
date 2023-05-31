/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef __PVGUIQT_PVEXPORTSELECTIONDLG_H__
#define __PVGUIQT_PVEXPORTSELECTIONDLG_H__

#include <QComboBox>
#include <QCheckBox>
#include <QRadioButton>

#include <pvkernel/rush/PVExporter.h>
#include <pvkernel/widgets/PVExportDlg.h>

namespace PVWidgets
{
class QKeySequenceWidget;
} // namespace PVWidgets

namespace Squey
{
class PVAxesCombination;
class PVView;
class PVSelection;
} // namespace Squey

namespace PVGuiQt
{

class PVAxesCombinationWidget;

/** Specific widget to export a selection to csv.
 *
 * Pop a file dialog to specify filename and the kind of export and do it.
 */
class PVExportSelectionDlg : public PVWidgets::PVExportDlg
{
	Q_OBJECT;

  public:
	/** Pop the FileDialog and perform the export.
	 *
	 * ExportSelectionDlg can't be created directly, creation and export are
	 * done in one step.
	 */
	static void export_selection(Squey::PVView& view, const Squey::PVSelection& sel);
	
	static void export_layers(Squey::PVView& view);

	/** Create a FileDialog to export selection
	 *
	 * @param custom_axes_combination : Combination of axis selected using the view menu
	 * @param view : The view to export
	 * @param parent : parent widget (as usual in Qt)
	 */
	PVExportSelectionDlg(
		Squey::PVView& view,
		QWidget* parent = 0,
		QFileDialog::AcceptMode accept_mode = QFileDialog::AcceptSave,
	    QFileDialog::FileMode file_mode = QFileDialog::AnyFile
	);

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
