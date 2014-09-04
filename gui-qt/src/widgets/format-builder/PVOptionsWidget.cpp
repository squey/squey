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
	_ignore_first_lines_spinbox = new QSpinBox();
	QLabel* first_lines_label = new QLabel("first line(s)");

	ignore_first_lines_layout->addWidget(ignore_label);
	ignore_first_lines_layout->addWidget(_ignore_first_lines_spinbox);
	ignore_first_lines_layout->addWidget(first_lines_label);
	ignore_first_lines_layout->addItem(new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	QHBoxLayout* last_line_layout = new QHBoxLayout();
	_last_line_checkbox = new QCheckBox();
	_last_line_label = new QLabel("Stop at line");
	_last_line_spinbox = new QSpinBox();
	_last_line_spinbox->setMaximum(CUSTOMER_LINESNUMBER);

	last_line_layout->addWidget(_last_line_checkbox);
	last_line_layout->addWidget(_last_line_label);
	last_line_layout->addWidget(_last_line_spinbox);
	last_line_layout->addItem(new QSpacerItem(1, 1, QSizePolicy::MinimumExpanding, QSizePolicy::Minimum));

	group_box_layout->addLayout(ignore_first_lines_layout);
	group_box_layout->addLayout(last_line_layout);
	group_box_layout->addItem(new QSpacerItem(100, 100, QSizePolicy::Minimum, QSizePolicy::MinimumExpanding));

	main_layout->addWidget(group_box);

	setLayout(main_layout);

	connect(_last_line_checkbox, SIGNAL(stateChanged(int)), this, SLOT(disable_specify_last_line(int)));
}

void PVInspector::PVOptionsWidget::set_lines_range(int first_line, int last_line)
{
	_last_line_spinbox->setValue(last_line);
	_ignore_first_lines_spinbox->setValue(first_line);

	disable_specify_last_line(last_line == 0 ? Qt::Unchecked : Qt::Checked);
}

void PVInspector::PVOptionsWidget::disable_specify_last_line(int checkstate)
{
	_last_line_label->setEnabled(checkstate == Qt::Checked);
	_last_line_spinbox->setEnabled(checkstate == Qt::Checked);
	_last_line_checkbox->setCheckState((Qt::CheckState)checkstate);
}
