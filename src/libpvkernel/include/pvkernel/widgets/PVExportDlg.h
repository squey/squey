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

#include <QDialog>
#include <QHash>
#include <QString>
#include <QStringList>

#include <utility>

#include <pvkernel/rush/PVExporter.h>
#include <pvkernel/widgets/PVExporterWidgetInterface.h>

class QComboBox;
class QStackedWidget;

namespace PVWidgets
{

/**
 * The format of an export and its options, asked for before the file it writes.
 *
 * The file is then asked for in the file dialog of the platform, which has no room for options.
 */
class PVExportDlg : public QDialog
{
	Q_OBJECT;

  public:
	/** What an export writes: a file, or a directory that receives a file per layer. */
	enum class Target { File, Directory };

	/**
	 * @param csv_widget The options of the CSV formats, which the dialog takes; nullptr for
	 *                   the default ones.
	 */
	explicit PVExportDlg(QWidget* parent = nullptr,
	                     Target target = Target::File,
	                     PVExporterWidgetInterface* csv_widget = nullptr);

	/** The file filters of the formats offered, such as ".csv.gz files (*.csv.gz)". */
	QStringList name_filters() const;
	QString selected_name_filter() const;
	/** Chooses the format of @p filter, as the user does. Does nothing if it is not offered. */
	void select_name_filter(QString const& filter);

	/** The extension of the chosen format, such as ".csv.gz". */
	QString file_extension() const;
	/** The exporter of the chosen format. */
	PVRush::PVExporterBase& exporter();

	/** The file or directory chosen after the options, empty until the dialog is accepted. */
	QString path() const { return _path; }

	/** Names the file the dialog suggests, before the extension of the format. */
	void set_file_name(QString const& name) { _file_name = name; }

  protected:
	/** Offers the format of @p filter ahead of the CSV ones, with @p page holding its options. */
	void add_format(QString const& filter, QWidget* page, PVRush::PVExporterBase& exporter);

  private:
	void insert_format(int index,
	                   QString const& filter,
	                   QWidget* page,
	                   PVRush::PVExporterBase& exporter);
	void choose_path();

	Target _target;
	QComboBox* _formats;
	QStackedWidget* _pages;
	/** The exporter of each format and the page of its options, by its filter. */
	QHash<QString, std::pair<PVRush::PVExporterBase*, QWidget*>> _exporters;
	/** How many formats add_format() put ahead of the CSV ones. */
	int _formats_ahead = 0;
	QString _file_name = "export";
	QString _path;
};
} // namespace PVWidgets

#endif // __PVKERNEL_WIDGETS_PVEXPORTDLG_H__
