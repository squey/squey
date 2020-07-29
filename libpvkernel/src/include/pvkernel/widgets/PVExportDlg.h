/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
