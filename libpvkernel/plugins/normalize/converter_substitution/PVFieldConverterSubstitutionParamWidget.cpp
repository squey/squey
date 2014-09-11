/**
 * \file PVFieldSubstitutionParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldConverterSubstitutionParamWidget.h"
#include "PVFieldConverterSubstitution.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QAction>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QCheckBox>

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::PVFieldConverterSubstitutionParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldConverterSubstitutionParamWidget::PVFieldConverterSubstitutionParamWidget() :
	PVFieldsConverterParamWidget(PVFilter::PVFieldsConverter_p(new PVFieldConverterSubstitution()))
{
	_action_menu = new QAction(QString("add Substitution"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldConverterSubstitutionParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldSubstitutionParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldConverterSubstitutionParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSubstitutionParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);

	QHBoxLayout* file_layout = new QHBoxLayout();

	QLabel* file_label = new QLabel("Conversion file:");
	_file_path_line_edit = new QLineEdit();
	_file_path_line_edit->setReadOnly(true);
	_file_path_line_edit->setText(args["path"].toString());

	QPushButton* browse_pushbutton = new QPushButton("...");

	file_layout->addWidget(file_label);
	file_layout->addWidget(_file_path_line_edit);
	file_layout->addWidget(browse_pushbutton);

	// Fields separator
	QHBoxLayout* fields_separator_layout = new QHBoxLayout();
	QLabel* separator_label = new QLabel(tr("Fields separator:"));
	_separator_char = new PVWidgets::QKeySequenceWidget();
	_separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_separator_char->setKeySequence(QKeySequence(args["sep"].toString()));
	_separator_char->setMaxNumKey(1);
	fields_separator_layout->addWidget(separator_label);
	fields_separator_layout->addWidget(_separator_char);
	fields_separator_layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// Quote character
	QHBoxLayout* quote_character_layout = new QHBoxLayout();
	QLabel* quote_label = new QLabel(tr("Quote character:"));
	_quote_char = new PVWidgets::QKeySequenceWidget();
	_quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_quote_char->setKeySequence(QKeySequence(args["quote"].toString()));
	_quote_char->setMaxNumKey(1);
	quote_character_layout->addWidget(quote_label);
	quote_character_layout->addWidget(_quote_char);
	quote_character_layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	QHBoxLayout* default_value_layout = new QHBoxLayout();

	_use_default_value_checkbox = new QCheckBox();
	_use_default_value_checkbox->setChecked(args["use_default_value"].toBool());
	QLabel* default_value_label = new QLabel("Default value:");
	_default_value_line_edit = new QLineEdit();
	_default_value_line_edit->setText(args["default_value"].toString());
	_default_value_line_edit->setEnabled(false);

	default_value_layout->addWidget(_use_default_value_checkbox);
	default_value_layout->addWidget(default_value_label);
	default_value_layout->addWidget(_default_value_line_edit);

	layout->addLayout(file_layout);
	layout->addLayout(fields_separator_layout);
	layout->addLayout(quote_character_layout);
	layout->addLayout(default_value_layout);
	layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	// Connections
	connect(browse_pushbutton, SIGNAL(clicked(bool)), this, SLOT(browse_conversion_file()));
	connect(_default_value_line_edit, SIGNAL(textChanged(QString)), this, SLOT(update_params()));
	connect(_use_default_value_checkbox, SIGNAL(stateChanged(int)), this, SLOT(use_default_value_checkbox_changed(int)));
	connect(_separator_char, SIGNAL(keySequenceChanged(const QKeySequence &)), this, SLOT(update_params()));
	connect(_quote_char, SIGNAL(keySequenceChanged(const QKeySequence &)), this, SLOT(update_params()));

	return _param_widget;
}

/******************************************************************************
 *
 * PVFilter::PVFieldConverterSubstitutionParamWidget::update_params
 *
 *****************************************************************************/
void PVFilter::PVFieldConverterSubstitutionParamWidget::update_params()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args["path"] = _file_path_line_edit->text();
	args["default_value"] = _default_value_line_edit->text();
	args["use_default_value"] = _use_default_value_checkbox->isChecked();
	args["sep"] = _separator_char->keySequence().toString();;
	args["quote"] = _quote_char->keySequence().toString();;

	get_filter()->set_args(args);
    emit args_changed_Signal();
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::browse_conversion_file()
{
	QFileDialog fd;

	QString filename = fd.getOpenFileName(nullptr, tr("Open File"), "", tr("Files (*.*)"));
	if (!filename.isEmpty()) {
		_file_path_line_edit->setText(filename);
	}

	update_params();
}

void PVFilter::PVFieldConverterSubstitutionParamWidget::use_default_value_checkbox_changed(int state)
{
	_default_value_line_edit->setEnabled(state == Qt::Checked);

	update_params();
}
