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

#ifndef PVOPTIONSWIDGET_H_
#define PVOPTIONSWIDGET_H_

#include <pvguiqt/PVLocalizedSpinBox.h>

#include <QWidget>
#include <QSpinBox>
#include <QCheckBox>
class QLabel;

namespace PVGuiQt
{
class PVPythonScriptWidget;
}

namespace PVInspector
{

class PVOptionsWidget : public QWidget
{
	Q_OBJECT

  public:
	PVOptionsWidget(QWidget* parent = nullptr);

  public:
	int first_line() { return _ignore_first_lines_spinbox->value(); }
	int line_count()
	{
		return _line_count_checkbox->checkState() == Qt::Checked ? _line_count_spinbox->value() : 0;
	}

	void set_lines_range(int first_line, int line_count);
	void set_python_script(const QString& python_script, bool is_path, bool disabled);

  Q_SIGNALS:
	void first_line_changed(int);
	void line_count_changed(int);
    void python_script_updated(const QString& python_script, bool is_path, bool disabled);

  private Q_SLOTS:
	void disable_specify_line_count(int checkstate);

  private:
	PVGuiQt::PVLocalizedSpinBox* _ignore_first_lines_spinbox;
	QLabel* _line_count_label;
	QCheckBox* _line_count_checkbox;
	PVGuiQt::PVLocalizedSpinBox* _line_count_spinbox;
	PVGuiQt::PVPythonScriptWidget* _python_scripting_widget;
};
}

#endif // PVOPTIONSWIDGET_H_
