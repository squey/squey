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

#ifndef PVMAPPINGPLOTTINGEDITDIALOG_H
#define PVMAPPINGPLOTTINGEDITDIALOG_H

#include <pvkernel/rush/PVFormat_types.h>

#include <QDialog>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QVBoxLayout>

class QScrollArea;
class QGroupBox;

namespace Squey
{
class PVMapped;
class PVPlotted;
} // namespace Squey

namespace PVWidgets
{

class PVMappingPlottingEditDialog : public QDialog
{
	Q_OBJECT
  public:
	PVMappingPlottingEditDialog(Squey::PVMapped* mapping,
	                            Squey::PVPlotted* plotting,
	                            QWidget* parent = nullptr);

  private:
	inline bool has_mapping() const { return _mapping != nullptr; };
	inline bool has_plotting() const { return _plotting != nullptr; }

	void init_layout();
	void finish_layout();
	void load_settings();
	void reset_settings_with_format();

	static QLabel* create_label(QString const& text, Qt::Alignment align = Qt::AlignCenter);

  private Q_SLOTS:
	void save_settings();

  private:
	QGridLayout* _main_grid;
	QVBoxLayout* _main_layout;
	QScrollArea* _main_scroll_area;
	Squey::PVMapped* _mapping;
	Squey::PVPlotted* _plotting;
	QList<PVRush::PVAxisFormat> const& _axes;
};
} // namespace PVWidgets

#endif
