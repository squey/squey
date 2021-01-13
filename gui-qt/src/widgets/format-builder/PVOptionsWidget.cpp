/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2020
 */

#include <pvbase/general.h>

#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>

#include <PVOptionsWidget.h>
#include <pvguiqt/PVPythonScriptWidget.h>

#include <iostream>

PVInspector::PVOptionsWidget::PVOptionsWidget(QWidget* parent /* = nullptr */) : QWidget(parent)
{
	auto main_layout = new QVBoxLayout();
	QGroupBox* lines_range_group_box = new QGroupBox(tr("Import lines range"));

	auto lines_range_group_box_layout = new QVBoxLayout(lines_range_group_box);

	auto ignore_first_lines_layout = new QHBoxLayout();

	QLabel* ignore_label = new QLabel("Ignore");
	_ignore_first_lines_spinbox = new PVGuiQt::PVLocalizedSpinBox();
	_ignore_first_lines_spinbox->setMaximum(std::numeric_limits<int>::max());
	QLabel* first_lines_label = new QLabel("first line(s) for each input file");

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
	connect(_python_scripting_widget, &PVGuiQt::PVPythonScriptWidget::python_script_updated, this, &PVInspector::PVOptionsWidget::python_script_updated);
	
	main_layout->addWidget(lines_range_group_box);
	main_layout->addWidget(_python_scripting_widget);
	main_layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Minimum, QSizePolicy::Expanding));

	setLayout(main_layout);

	connect(_line_count_spinbox, static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged),
	        this, [&]() { Q_EMIT line_count_changed(line_count()); });
	connect(_ignore_first_lines_spinbox,
	        static_cast<void (QSpinBox::*)(int)>(&QSpinBox::valueChanged), this,
	        &PVOptionsWidget::first_line_changed);
	connect(_line_count_checkbox, &QCheckBox::stateChanged, this,
	        &PVOptionsWidget::disable_specify_line_count);
}

void PVInspector::PVOptionsWidget::set_lines_range(int first_line, int line_count)
{
	_line_count_spinbox->setValue(line_count);
	_ignore_first_lines_spinbox->setValue(first_line);

	disable_specify_line_count(line_count == 0 ? Qt::Unchecked : Qt::Checked);
}

void PVInspector::PVOptionsWidget::disable_specify_line_count(int checkstate)
{
	_line_count_label->setEnabled(checkstate == Qt::Checked);
	_line_count_spinbox->setEnabled(checkstate == Qt::Checked);
	_line_count_checkbox->setCheckState((Qt::CheckState)checkstate);

	Q_EMIT line_count_changed(line_count());
}

void PVInspector::PVOptionsWidget::set_python_script(const QString& python_script, bool is_path, bool disabled)
{
	_python_scripting_widget->set_python_script(python_script, is_path, disabled);
}
