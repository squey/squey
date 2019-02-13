/**
 * @file
 *
 * @copyright (C) Picviz Labs 2014-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include "PVFieldSplitterKeyValueParamWidget.h"
#include "PVFieldSplitterKeyValue.h"

#include <pvkernel/filter/PVFieldsFilter.h>

#include <QAction>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QInputDialog>
#include <QMessageBox>
#include <QApplication>
#include <QClipboard>

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget
 *
 *****************************************************************************/
PVFilter::PVFieldSplitterKeyValueParamWidget::PVFieldSplitterKeyValueParamWidget()
    : PVFieldsSplitterParamWidget(PVFilter::PVFieldsSplitter_p(new PVFieldSplitterKeyValue()))
{
}

/******************************************************************************
 *
 * PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu
 *
 *****************************************************************************/
QAction* PVFilter::PVFieldSplitterKeyValueParamWidget::get_action_menu(QWidget* parent)
{
	return new QAction(QString("add Key-Value Splitter"), parent);
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
	quote_character_layout->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	structure_layout->addLayout(fields_separator_layout);
	structure_layout->addLayout(affectation_operator_layout);
	structure_layout->addLayout(quote_character_layout);
	structure_layout->addSpacerItem(
	    new QSpacerItem(1, 1, QSizePolicy::Expanding, QSizePolicy::Expanding));

	QHBoxLayout* keys_layout = new QHBoxLayout();
	QVBoxLayout* buttons_layout = new QVBoxLayout();

	QPushButton* add_button = new QPushButton(tr("Add"));
	_del_button = new QPushButton(tr("Delete"));
	_up_button = new QPushButton(tr("Move up"));
	_down_button = new QPushButton(tr("Move down"));
	_copy_button = new QPushButton(tr("Copy"));
	QPushButton* paste_button = new QPushButton(tr("Paste"));

	add_button->setIcon(QIcon(":/document-new"));
	_del_button->setIcon(QIcon(":/red-cross"));
	_up_button->setIcon(QIcon(":/go-up"));
	_down_button->setIcon(QIcon(":/go-down"));
	_copy_button->setIcon(QIcon(":/edit-copy"));
	paste_button->setIcon(QIcon(":/edit-paste"));

	buttons_layout->addWidget(add_button);
	buttons_layout->addWidget(_del_button);
	buttons_layout->addWidget(_up_button);
	buttons_layout->addWidget(_down_button);
	buttons_layout->addWidget(_copy_button);
	buttons_layout->addWidget(paste_button);

	QGroupBox* keys_groupbox = new QGroupBox("Keys");
	keys_groupbox->setLayout(keys_layout);

	_keys_list = new QListWidget();
	_keys_list->addItems(args["keys"].toStringList());

	// Make each item editable
	for (int i = 0; i < _keys_list->count(); i++) {
		_keys_list->item(i)->setFlags(_keys_list->item(i)->flags() | Qt::ItemIsEditable);
	}
	_keys_list->setEditTriggers(QAbstractItemView::DoubleClicked |
	                            QAbstractItemView::EditKeyPressed);
	_keys_list->setAlternatingRowColors(true);

	// selection mode
	_keys_list->setSelectionMode(QAbstractItemView::ExtendedSelection);

	keys_layout->addWidget(_keys_list);
	keys_layout->addLayout(buttons_layout);

	layout->addWidget(structure_groupbox);
	layout->addWidget(keys_groupbox);

	connect(_affectation_operator_lineedit, &QLineEdit::textChanged, this,
	        &PVFieldSplitterKeyValueParamWidget::update_params);
	connect(_separator_char_lineedit, &QLineEdit::textChanged, this,
	        &PVFieldSplitterKeyValueParamWidget::update_params);
	connect(_quote_char, &PVWidgets::QKeySequenceWidget::keySequenceChanged, this,
	        &PVFieldSplitterKeyValueParamWidget::update_params);
	connect(add_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::add_new_key);
	connect(_del_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::del_keys);
	connect(_up_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::move_key_up);
	connect(_down_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::move_key_down);
	connect(_copy_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::copy_keys);
	connect(paste_button, &QAbstractButton::clicked, this,
	        &PVFieldSplitterKeyValueParamWidget::paste_keys);
	connect(_keys_list, &QListWidget::itemSelectionChanged, this,
	        &PVFieldSplitterKeyValueParamWidget::selection_has_changed);

	selection_has_changed();

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
	for (int i = 0; i < _keys_list->count(); i++) {
		keys << _keys_list->item(i)->text();
	}
	args["keys"] = QVariant(keys);

	get_filter()->set_args(args);
	Q_EMIT args_changed_Signal();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::add_new_keys(QStringList& keys)
{
	QStringList keys_found_list;

	for (QString key : keys) {
		// Look if the value already exist
		QList<QListWidgetItem*> items = _keys_list->findItems(key, Qt::MatchExactly);
		if (items.count() == 0) {
			if (!key.isEmpty()) {
				QListWidgetItem* new_item = new QListWidgetItem(key);
				new_item->setFlags(new_item->flags() | Qt::ItemIsEditable);
				_keys_list->addItem(new_item);
			}
		} else {
			keys_found_list << key;
		}
	}
	update_children_count();

	if (!keys_found_list.isEmpty()) {
		QMessageBox(QMessageBox::Warning, tr("This key(s) already exist."),
		            tr("This key(s) '%1' already exist!").arg(keys_found_list.join(", ")),
		            QMessageBox::Ok)
		    .exec();
	}
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::add_new_key()
{
	bool ok;
	QString key =
	    QInputDialog::getText(nullptr, tr("Enter new key"), tr("Key:"), QLineEdit::Normal, "", &ok);

	QStringList keys = QStringList(key);

	add_new_keys(keys);
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::del_keys()
{
	QList<QListWidgetItem*> keys = _keys_list->selectedItems();
	for (QListWidgetItem* key : keys) {
		delete _keys_list->takeItem(_keys_list->row(key));
	}

	update_children_count();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::update_children_count()
{
	PVCore::PVArgumentList args = get_filter()->get_args();

	QStringList keys;
	for (int i = 0; i < _keys_list->count(); i++) {
		keys << _keys_list->item(i)->text();
	}
	args["keys"] = QVariant(keys);

	get_filter()->set_args(args);
	set_child_count(keys.size());
	Q_EMIT args_changed_Signal();
	Q_EMIT nchilds_changed_Signal();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::move_key_down()
{
	QList<QListWidgetItem*> keys = _keys_list->selectedItems();

	if (keys.isEmpty()) {
		return;
	}

	/* keys must be necessarily ordered (decreasing order of their indices in the QListWidget) to
	 * make the move down work.
	 */
	qSort(keys.begin(), keys.end(), [&](const QListWidgetItem* i1, const QListWidgetItem* i2) {
		return _keys_list->row(i1) > _keys_list->row(i2);
	});

	if (_keys_list->row(keys.last()) < (_keys_list->count() - 1)) {
		for (QListWidgetItem* key : keys) {
			int currentIndex = _keys_list->row(key);
			QListWidgetItem* currentItem = _keys_list->takeItem(currentIndex);
			_keys_list->insertItem(currentIndex + 1, currentItem);
			_keys_list->setCurrentItem(currentItem, QItemSelectionModel::Select);
		}
	}
	update_children_count();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::move_key_up()
{
	QList<QListWidgetItem*> keys = _keys_list->selectedItems();

	if (keys.isEmpty()) {
		return;
	}

	/* keys must be necessarily ordered (increasing order of their indices in the QListWidget) to
	 * make the move up work.
	 */
	qSort(keys.begin(), keys.end(), [&](const QListWidgetItem* i1, const QListWidgetItem* i2) {
		return _keys_list->row(i1) < _keys_list->row(i2);
	});

	if (_keys_list->row(keys.first()) > 0) {
		for (QListWidgetItem* key : keys) {
			int currentIndex = _keys_list->row(key);
			QListWidgetItem* currentItem = _keys_list->takeItem(currentIndex);
			_keys_list->insertItem(currentIndex - 1, currentItem);
			_keys_list->setCurrentItem(currentItem, QItemSelectionModel::Select);
		}
	}
	update_children_count();
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::copy_keys()
{
	QStringList strings_list;
	for (QListWidgetItem* item : _keys_list->selectedItems())
		strings_list << item->text();

	QApplication::clipboard()->setText(strings_list.join("\n"));
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::paste_keys()
{
	QString strings = QApplication::clipboard()->text();
	// Users can paste text from the clipboard regardless of whether it contains whitespace
	// characters like '\t', '\n', '\v', '\f', '\r', and ' '.
	strings = strings.simplified();
	QStringList strings_list = strings.split(" ");
	add_new_keys(strings_list);
}

void PVFilter::PVFieldSplitterKeyValueParamWidget::selection_has_changed()
{
	QList<QListWidgetItem*> keys = _keys_list->selectedItems();

	if (keys.isEmpty()) {
		_del_button->setDisabled(true);
		_up_button->setDisabled(true);
		_down_button->setDisabled(true);
		_copy_button->setDisabled(true);

		return;
	}

	/* must sort keys to make first() have the smallest index and last() have the greatest index
	 */
	qSort(keys.begin(), keys.end(), [&](const QListWidgetItem* i1, const QListWidgetItem* i2) {
		return _keys_list->row(i1) < _keys_list->row(i2);
	});

	_up_button->setEnabled(_keys_list->row(keys.first()) > 0);
	_down_button->setEnabled(_keys_list->row(keys.last()) < (_keys_list->count() - 1));

	_del_button->setEnabled(true);
	_copy_button->setEnabled(true);
}
