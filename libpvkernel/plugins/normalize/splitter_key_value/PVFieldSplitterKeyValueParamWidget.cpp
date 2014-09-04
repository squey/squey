/**
 * \file PVFieldSplitterKeyValueParamWidget.cpp
 *
 * Copyright (C) Picviz Labs 2014
 */

#include "PVFieldSplitterKeyValueParamWidget.h"
#include "PVFieldSplitterKeyValue.h"

#include <pvkernel/filter/PVFieldsFilter.h>
#include <pvkernel/filter/PVElementFilterByFields.h>

#include <QAction>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QInputDialog>
#include <QFileDialog>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget() :
	PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterKeyValue()))
{
	_action_menu = new QAction(QString("add Key Value"), NULL);
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu()
{
    PVLOG_DEBUG("get action PVFieldSplitterKeyValueParamWidget\n");
    assert(_action_menu);
    return _action_menu;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget
 *
 *****************************************************************************/
QWidget* PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget()
{
	PVLOG_DEBUG("PVFilter::PVFieldSplitterKeyValueParamWidget::get_param_widget()     start\n");

	PVCore::PVArgumentList args = get_filter()->get_args();

	_param_widget = new QWidget();

	QVBoxLayout* layout = new QVBoxLayout(_param_widget);

	QVBoxLayout* structure_layout = new QVBoxLayout();

	QGroupBox* structure_groupbox = new QGroupBox("Structure");
	structure_groupbox->setLayout(structure_layout);

	// Fields separator
	QHBoxLayout* fields_separator_layout = new QHBoxLayout();
	QLabel* separator_label = new QLabel(tr("Fields separator:"));
	_separator_char_lineedit = new QLineEdit();
	_separator_char_lineedit->setText(args["sep"].toString());
	fields_separator_layout->addWidget(separator_label);
	fields_separator_layout->addWidget(_separator_char_lineedit);

	// Affectation operator
	QHBoxLayout* affectation_operator_layout = new QHBoxLayout();
	QLabel* affectation_operator_label = new QLabel("Affectation operator:");
	_affectation_operator_lineedit = new QLineEdit();
	_affectation_operator_lineedit->setText(args["affectation"].toString());
	affectation_operator_layout->addWidget(affectation_operator_label);
	affectation_operator_layout->addWidget(_affectation_operator_lineedit);

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

	structure_layout->addLayout(fields_separator_layout);
	structure_layout->addLayout(affectation_operator_layout);
	structure_layout->addLayout(quote_character_layout);
	structure_layout->addSpacerItem(new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	QHBoxLayout* keys_layout = new QHBoxLayout();
	QVBoxLayout* buttons_layout = new QVBoxLayout();

	QPushButton* add_button = new QPushButton("+");
	QPushButton* del_button = new QPushButton("-");
	QPushButton* up_button = new QPushButton("^");
	QPushButton* down_button = new QPushButton("v");

	buttons_layout->addWidget(add_button);
	buttons_layout->addWidget(del_button);
	buttons_layout->addWidget(up_button);
	buttons_layout->addWidget(down_button);

	QGroupBox* keys_groupbox = new QGroupBox("Keys");
	keys_groupbox->setLayout(keys_layout);

	_keys_list = new QListWidget();
	_keys_list->addItems(args["keys"].toStringList());
	_keys_list->setSelectionMode(QAbstractItemView::MultiSelection);
	keys_layout->addWidget(_keys_list);
	keys_layout->addLayout(buttons_layout);

	layout->addWidget(structure_groupbox);
	layout->addWidget(keys_groupbox);

	connect(_affectation_operator_lineedit, SIGNAL(textChanged(const QString &)), this, SLOT(update_params()));
	connect(_separator_char_lineedit, SIGNAL(textChanged(const QString &)), this, SLOT(update_params()));
	connect(_quote_char, SIGNAL(keySequenceChanged(const QKeySequence &)), this, SLOT(update_params()));
	connect(add_button, SIGNAL(clicked(bool)), this, SLOT(add_new_key()));
	connect(del_button, SIGNAL(clicked(bool)), this, SLOT(del_keys()));
	connect(up_button, SIGNAL(clicked(bool)), this, SLOT(move_key_up()));
	connect(down_button, SIGNAL(clicked(bool)), this, SLOT(move_key_down()));

	return _param_widget;
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::update_params
 *
 *****************************************************************************/
void PVFilter::PVFieldSplitterKeyValueParamWidget::update_params()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	args["sep"] = _separator_char_lineedit->text();
	args["quote"] = _quote_char->keySequence().toString();
	args["affectation"] = _affectation_operator_lineedit->text();

	QStringList keys;
	for (int i = 0 ; i < _keys_list->count() ; i++) {
		keys << _keys_list->item(i)->text();
	}
	args["keys"] = QVariant(keys);

	get_filter()->set_args(args);
    emit args_changed_Signal();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::add_new_key()
{
	bool ok;
	QString key = QInputDialog::getText(nullptr, tr("Enter new key"), tr("Key:"), QLineEdit::Normal, "", &ok);

	if (!key.isEmpty()) {
		_keys_list->addItem(new QListWidgetItem(key));
		update_children_count();
	}
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::del_keys()
{
	QList<QListWidgetItem *> keys = _keys_list->selectedItems();
	for (QListWidgetItem* key : keys) {
		delete _keys_list->takeItem(_keys_list->row(key));
	}

	update_children_count();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::update_children_count()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	QStringList keys;
	for (int i = 0 ; i < _keys_list->count() ; i++) {
		keys << _keys_list->item(i)->text();
	}
	args["keys"] = QVariant(keys);

	get_filter()->set_args(args);
	set_child_count(keys.size());
	emit args_changed_Signal();
	emit nchilds_changed_Signal();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::move_key_down()
{
	int currentIndex = _keys_list->currentRow();
	QListWidgetItem *currentItem = _keys_list->takeItem(currentIndex);
	_keys_list->insertItem(currentIndex+1, currentItem);
	_keys_list->setCurrentRow(currentIndex+1);
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::move_key_up()
{
	int currentIndex = _keys_list->currentRow();
	QListWidgetItem *currentItem = _keys_list->takeItem(currentIndex);
	_keys_list->insertItem(currentIndex-1, currentItem);
	_keys_list->setCurrentRow(currentIndex-1);
}

