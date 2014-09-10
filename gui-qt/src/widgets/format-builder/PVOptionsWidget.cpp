/**
 * \file PVOptionsWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include <QGroupBox>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QSpacerItem>

#include <pvkernel/core/general.h>
#include <PVOptionsWidget.h>

#include <iostream>

PVInspector::PVOptionsWidget::PVOptionsWidget(QWidget* parent /* = nullptr */) : QWidget(parent)
{
	QVBoxLayout* main_layout = new QVBoxLayout();
	QGroupBox* group_box = new QGroupBox(tr("Import lines range"));

	QVBoxLayout* group_box_layout = new QVBoxLayout(group_box);

	QHBoxLayout* ignore_first_lines_layout = new QHBoxLayout();

	QLabel* ignore_label = new QLabel("Ignore");
	_ignore_first_lines_spinbox = new PVGuiQt::PVLocalizedSpinBox();
	_ignore_first_lines_spinbox->setMaximum(PICVIZ_LINES_MAX);
	QLabel* first_lines_label = new QLabel("first line(s) for each input file");

	ignore_first_lines_layout->addWidget(ignore_label);
	ignore_first_lines_layout->addWidget(_ignore_first_lines_spinbox);
	ignore_first_lines_layout->addWidget(first_lines_label);
	ignore_first_lines_layout->addItem(new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	QHBoxLayout* line_count_layout = new QHBoxLayout();
	_line_count_checkbox = new QCheckBox();
	_line_count_label = new QLabel("Stop at line count");
	_line_count_spinbox = new PVGuiQt::PVLocalizedSpinBox();
	_line_count_spinbox->setMaximum(PICVIZ_LINES_MAX);

	line_count_layout->addWidget(_line_count_checkbox);
	line_count_layout->addWidget(_line_count_label);
	line_count_layout->addWidget(_line_count_spinbox);
	line_count_layout->addItem(new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	group_box_layout->addLayout(ignore_first_lines_layout);
	group_box_layout->addLayout(line_count_layout);
	group_box_layout->addItem(new QSpacerItem(100, 100, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding));

	main_layout->addWidget(group_box);

	setLayout(main_layout);

	connect(_line_count_checkbox, SIGNAL(stateChanged(int)), this, SLOT(disable_specify_line_count(int)));
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
}
