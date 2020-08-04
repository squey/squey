/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
