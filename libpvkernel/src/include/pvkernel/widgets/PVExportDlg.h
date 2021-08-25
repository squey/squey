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

#ifndef __PVKERNEL_WIDGETS_PVEXPORTDLG_H__
#define __PVKERNEL_WIDGETS_PVEXPORTDLG_H__

#include <QComboBox>
#include <QCheckBox>
#include <QRadioButton>

#include <pvkernel/rush/PVExporter.h>
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/widgets/PVExporterWidgetInterface.h>

class QGroupBox;

namespace PVWidgets
{
class QKeySequenceWidget;
} // namespace PVWidgets

namespace PVWidgets
{

/** Specific widget to export a selection to csv.
 *
 * Pop a file dialog to specify filename and the kind of export and do it.
 */
class PVExportDlg : public PVWidgets::PVFileDialog
{
	Q_OBJECT;

	/** Create a FileDialog to export selection
	 *
	 * @param parent : parent widget (as usual in Qt)
	 */
  public:
	PVExportDlg(
		QWidget* parent = 0,
		QFileDialog::AcceptMode accept_mode = QFileDialog::AcceptSave,
	    QFileDialog::FileMode file_mode = QFileDialog::AnyFile
	);

	PVWidgets::PVExporterWidgetInterface* exporter_widget() { return _exporter_widget; }

	QString file_extension() const;

  protected:
	QStringList _name_filters;
	QGroupBox* _groupbox;
	std::function<void(const QString& filter)> _filter_selected_f;
	QMetaObject::Connection _conn;
	PVWidgets::PVExporterWidgetInterface* _exporter_widget = nullptr;
};
} // namespace PVWidgets

#endif // __PVKERNEL_WIDGETS_PVEXPORTDLG_H__
