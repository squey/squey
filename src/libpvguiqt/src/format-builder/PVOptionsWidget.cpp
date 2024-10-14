//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvbase/general.h>

#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>

#include <PVOptionsWidget.h>
#include <pvguiqt/PVPythonScriptWidget.h>

#include <iostream>

App::PVOptionsWidget::PVOptionsWidget(QWidget* parent /* = nullptr */) : QWidget(parent)
{
	auto main_layout = new QVBoxLayout();
	auto* lines_range_group_box = new QGroupBox(tr("Import lines range"));

	auto lines_range_group_box_layout = new QVBoxLayout(lines_range_group_box);

	auto ignore_first_lines_layout = new QHBoxLayout();

	auto* ignore_label = new QLabel("Ignore");
	_ignore_first_lines_spinbox = new PVGuiQt::PVLocalizedSpinBox();
	_ignore_first_lines_spinbox->setMaximum(std::numeric_limits<int>::max());
	auto* first_lines_label = new QLabel("first line(s) for each input file");

	ignore_first_lines_layout->addWidget(ignore_label);
	ignore_first_lines_layout->addWidget(_ignore_first_lines_spinbox);
	ignore_first_lines_layout->addWidget(first_lines_label);
	ignore_first_lines_layout->addItem(
	    new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	auto line_count_layout = new QHBoxLayout();
	_line_count_checkbox = new QCheckBox();
	_line_count_label = new QLabel("Stop at line count");
	_line_count_spinbox = new PVGuiQt::PVLocalizedSpinBox();
	_line_count_spinbox->setMaximum(std::numeric_limits<int>::max());

	line_count_layout->addWidget(_line_count_checkbox);
	line_count_layout->addWidget(_line_count_label);
	line_count_layout->addWidget(_line_count_spinbox);
	line_count_layout->addItem(
	    new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	lines_range_group_box_layout->addLayout(ignore_first_lines_layout);
	lines_range_group_box_layout->addLayout(line_count_layout);
	lines_range_group_box_layout->addItem(
	    new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	_python_scripting_widget = new PVGuiQt::PVPythonScriptWidget(this);
	connect(_python_scripting_widget, &PVGuiQt::PVPythonScriptWidget::python_script_updated, this, &App::PVOptionsWidget::python_script_updated);
	
	main_layout->addWidget(lines_range_group_box);
	main_layout->addWidget(_python_scripting_widget);
	_python_scripting_widget->setSizePolicy(QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));

	setLayout(main_layout);

	connect(_line_count_spinbox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
	        this, [&]() { Q_EMIT line_count_changed(line_count()); });
	connect(_ignore_first_lines_spinbox,
	        static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
	        &PVOptionsWidget::first_line_changed);
	connect(_line_count_checkbox, &QCheckBox::checkStateChanged, this,
	        &PVOptionsWidget::disable_specify_line_count);
}

void App::PVOptionsWidget::set_lines_range(int first_line, int line_count)
{
	_line_count_spinbox->setValue(line_count);
	_ignore_first_lines_spinbox->setValue(first_line);

	disable_specify_line_count(line_count == 0 ? Qt::Unchecked : Qt::Checked);
}

void App::PVOptionsWidget::disable_specify_line_count(int checkstate)
{
	_line_count_label->setEnabled(checkstate == Qt::Checked);
	_line_count_spinbox->setEnabled(checkstate == Qt::Checked);
	_line_count_checkbox->setCheckState((Qt::CheckState)checkstate);

	Q_EMIT line_count_changed(line_count());
}

void App::PVOptionsWidget::set_python_script(const QString& python_script, bool is_path, bool disabled)
{
	_python_scripting_widget->set_python_script(python_script, is_path, disabled);
}
