/**
 * \file PVExportSelectionDlg.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QLabel>
#include <QSpacerItem>

#include <pvguiqt/PVExportSelectionDlg.h>
#include <pvguiqt/PVAxesCombinationWidget.h>

PVGuiQt::PVExportSelectionDlg::PVExportSelectionDlg(Picviz::PVAxesCombination& custom_axes_combination, Picviz::PVView* view, QWidget* parent /* = 0 */) :
	QFileDialog(parent), _custom_axes_combination(custom_axes_combination)
{
	setWindowTitle(tr("Export selection"));
	setAcceptMode(QFileDialog::AcceptSave);

	QGridLayout* main_layout = (QGridLayout *) layout();

	QGroupBox* group_box = new QGroupBox();
	main_layout->addWidget(group_box, 6, 1, 1, 2);

	QVBoxLayout* export_layout = new QVBoxLayout();
	group_box->setLayout(export_layout);

	// Export column name
	_columns_header = new QCheckBox("Export column names as header");
	_columns_header->setCheckState(Qt::CheckState::Checked);
	export_layout->addWidget(_columns_header, 2, 0);

	// Separator character
	QHBoxLayout* fields_separator_layout = new QHBoxLayout();
	QLabel* separator_label = new QLabel(tr("Fields separator:"));
	_separator_char = new PVWidgets::QKeySequenceWidget();
	_separator_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_separator_char->setKeySequence(QKeySequence(","));
	_separator_char->setMaxNumKey(1);
	fields_separator_layout->addWidget(separator_label);
	fields_separator_layout->addWidget(_separator_char);
	fields_separator_layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
	export_layout->addLayout(fields_separator_layout);

	// Quote character
	QHBoxLayout* quote_character_layout = new QHBoxLayout();
	QLabel* quote_label = new QLabel(tr("Quote character:"));
	_quote_char = new PVWidgets::QKeySequenceWidget();
	_quote_char->setClearButtonShow(PVWidgets::QKeySequenceWidget::NoShow);
	_quote_char->setKeySequence(QKeySequence("\""));
	_quote_char->setMaxNumKey(1);
	quote_character_layout->addWidget(quote_label);
	quote_character_layout->addWidget(_quote_char);
	quote_character_layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
	export_layout->addLayout(quote_character_layout);

	// Use current axes combination
	_use_current_axes_combination = new QRadioButton("Use current axes combination");
	_use_current_axes_combination->setChecked(true);
	export_layout->addWidget(_use_current_axes_combination, 3, 0);

	// Use custom axes combination
	QHBoxLayout* custom_axes_combination_layout = new QHBoxLayout();
	QRadioButton* use_custom_axes_combination = new QRadioButton("Use custom axes combination");
	custom_axes_combination_layout->addWidget(use_custom_axes_combination);
	_edit_axes_combination = new QPushButton("Edit");
	_edit_axes_combination->setEnabled(use_custom_axes_combination->isChecked());
	custom_axes_combination_layout->addWidget(_edit_axes_combination);
	custom_axes_combination_layout->addSpacerItem(new QSpacerItem(1,1,QSizePolicy::Expanding, QSizePolicy::Expanding));
	export_layout->addLayout(custom_axes_combination_layout);

	connect(use_custom_axes_combination, SIGNAL(toggled(bool)), this, SLOT(show_axes_combination_widget(bool)));
	connect(_edit_axes_combination, SIGNAL(clicked()), this, SLOT(edit_axes_combination()));

	_axes_combination_widget = new PVAxesCombinationWidget(_custom_axes_combination, view);
	_axes_combination_widget->setWindowModality(Qt::WindowModal);
	_axes_combination_widget->hide();
}

void PVGuiQt::PVExportSelectionDlg::show_axes_combination_widget(bool show)
{
	_edit_axes_combination->setEnabled(show);
}

void PVGuiQt::PVExportSelectionDlg::edit_axes_combination()
{
	_axes_combination_widget->show();

}
