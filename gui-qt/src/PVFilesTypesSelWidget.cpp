/**
 * \file PVFilesTypesSelWidget.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QListWidget>

#include <PVMainWindow.h>

#include <pvkernel/core/general.h>

#include <PVFilesTypesSelWidget.h>

// Model
PVInspector::PVFilesTypesSelModel::PVFilesTypesSelModel(map_files_types& files_types, QObject* parent):
	QAbstractTableModel(parent),
	_files_types(files_types)
{
	_header_name[0] = "File path";
	_header_name[1] = "File types";
	_org_files_types = files_types;
}

int PVInspector::PVFilesTypesSelModel::rowCount(const QModelIndex &parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (parent.isValid())
		return 0;

	return _files_types.size();
}

int PVInspector::PVFilesTypesSelModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (parent.isValid())
		return 0;

	return 2;
}

QVariant PVInspector::PVFilesTypesSelModel::data(const QModelIndex& index, int role) const
{
	if (role != Qt::DisplayRole && role != Qt::EditRole)
		return QVariant();
	map_files_types::const_iterator it = _files_types.begin();
	std::advance(it, index.row());
	if (index.column() == 0)
		return (*it).first;

	if (role == Qt::DisplayRole)
		return (*it).second.join("\n");

	// Provide a list with two QStringList: the first one correspond to the selected items, and the second one to the original ones
	QList<QVariant> ret;
	ret << (*it).second;

	it = _org_files_types.begin();
	std::advance(it, index.row());
	ret << (*it).second;

	return ret;
}

bool PVInspector::PVFilesTypesSelModel::setData(const QModelIndex& index, const QVariant &value, int role)
{
	if (index.column() != 1 || role != Qt::EditRole)
		return false; // File name are not editable !

	map_files_types::iterator it = _files_types.begin();
	std::advance(it, index.row());
	if (it == _files_types.end())
		return false; // Should never happen !

	(*it).second = value.toStringList();

	emit dataChanged(index, index);

	return true;
}

Qt::ItemFlags PVInspector::PVFilesTypesSelModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret = Qt::ItemIsEnabled;
	if (index.column() == 1)
		ret |= Qt::ItemIsEditable;
	return ret;
}

QVariant PVInspector::PVFilesTypesSelModel::headerData(int section, Qt::Orientation orientation, int role) const
{
	if (orientation != Qt::Horizontal || section >= 2 || role != Qt::DisplayRole)
		return QAbstractTableModel::headerData(section, orientation, role);
	return _header_name[section];
}

void PVInspector::PVFilesTypesSelModel::emitAllTypesChanged()
{
	emit dataChanged(QAbstractTableModel::index(1,0), QAbstractTableModel::index(1,_files_types.size()-1));
}

// Delegate
// Show a combo box on the second column

PVInspector::PVFilesTypesSelDelegate::PVFilesTypesSelDelegate(QObject* parent) :
	QStyledItemDelegate(parent)
{
}

QWidget* PVInspector::PVFilesTypesSelDelegate::createEditor(QWidget *parent, const QStyleOptionViewItem& /*option*/, const QModelIndex& /*index*/) const
{
	QListWidget* editor = new QListWidget(parent);
	editor->setSelectionMode(QAbstractItemView::ExtendedSelection);
	return editor;
}

void PVInspector::PVFilesTypesSelDelegate::setEditorData(QWidget* editor, const QModelIndex& index) const
{
	QList<QVariant> model_data = index.model()->data(index, Qt::EditRole).toList();
	QStringList sel_list = model_data[0].toStringList();
	QStringList org_list = model_data[1].toStringList();

	QListWidget* listBox = static_cast<QListWidget*>(editor);
	listBox->clear();

	// Insert items
	QListWidgetItem *item;
	for (int i = 0; i < org_list.size(); i++) {
		QString const& f = org_list[i];
		bool issel = sel_list.contains(f);
		item = new QListWidgetItem(f);
		listBox->insertItem(-1, item);
		item->setSelected(issel);
	}
}

void PVInspector::PVFilesTypesSelDelegate::setModelData(QWidget *editor, QAbstractItemModel *model, const QModelIndex &index) const
{
	QStringList type_list;

	QListWidget* listBox = static_cast<QListWidget*>(editor);
	// Get the types selected
	for (int i = 0; i < listBox->count(); i++) {
		QListWidgetItem* item = listBox->item(i);
		if (item->isSelected()) {
			type_list << item->text();
		}
	}
	model->setData(index, type_list, Qt::EditRole);
}

void PVInspector::PVFilesTypesSelDelegate::updateEditorGeometry(QWidget* editor, const QStyleOptionViewItem &option, const QModelIndex& /*index*/) const
{
	editor->setGeometry(option.rect);
}

QSize PVInspector::PVFilesTypesSelDelegate::sizeHint(const QStyleOptionViewItem & option, const QModelIndex & index) const
{
	QSize ret = QStyledItemDelegate::sizeHint(option, index);
	ret.setWidth(ret.width()*1.2);
	ret.setHeight(ret.height()*1.2);
	return ret;
}


// Actual widget

PVInspector::PVFilesTypesSelWidget::PVFilesTypesSelWidget(PVInspector::PVMainWindow *parent, map_files_types& files_types):
	QDialog(parent),
	_files_types(files_types)
{
	main_window = parent;

	// Initalise layouts
	QVBoxLayout *main_layout = new QVBoxLayout();
	QVBoxLayout *all_types_layout = new QVBoxLayout();
	QHBoxLayout *btn_layout = new QHBoxLayout();

	// Files -> types
	_files_types_view = new QTableView();
	_files_types_model = new PVFilesTypesSelModel(_files_types);
	_files_types_del = new PVFilesTypesSelDelegate();
	_files_types_view->setModel(_files_types_model);
	_files_types_view->setItemDelegateForColumn(1, _files_types_del);
	_files_types_view->resizeRowsToContents();
	_files_types_view->resizeColumnsToContents();


	// Types for everyone
	_all_types_check = new QCheckBox("Set types for every files:");
	_all_types_list = new QListWidget();
	_all_types_list->setSelectionMode(QAbstractItemView::ExtendedSelection);
	_all_types_list->setSizePolicy(QSizePolicy(QSizePolicy::Maximum, QSizePolicy::MinimumExpanding));
	_all_types_list->setMaximumHeight(100);

	// Compute the union and intersection of types
	QStringList types_union;
	QStringList types_intersec = (*files_types.begin()).second;

	map_files_types::iterator it;
	for (it = files_types.begin(); it != files_types.end(); it++) {
		QStringList& types = (*it).second;

		// Union
#if (QT_VERSION >= 0x040700) // QList<T>::reserve has been introduced in QT 4.7
		if (types_union.size() < types.size()) {
			types_union.reserve(types.size());
		}
#endif
		QStringList::iterator it_t = types.begin();
		for (; it_t != types.end(); it_t++) {
			if (!types_union.contains(*it_t)) {
				types_union << *it_t;
			}
		}

		// Intersec
		it_t = types_intersec.begin();
	   	while (it_t != types_intersec.end()) {
			if (!types.contains(*it_t)) {
				QStringList::iterator it_rem = it_t;
				it_t++;
				bool was_last = it_t == types_intersec.end();
				types_intersec.erase(it_rem);
				if (was_last) {
					break;
				}
			}
			else {
				it_t++;
			}
		}
	}

	// Set the intersection as the default selection
	// and set a gray background for the union
	
	QListWidgetItem* item;
	QStringList::const_iterator it_sl;
	for (it_sl = types_union.begin(); it_sl != types_union.end(); it_sl++) {
		QString const& typen = *it_sl;
		item = new QListWidgetItem(typen);
		_all_types_list->insertItem(-1, item);
		if (types_intersec.contains(typen))
			item->setSelected(true);
		else
			item->setBackgroundColor(Qt::lightGray);
	}
	_all_types_list->setEnabled(false);

	all_types_layout->addWidget(_all_types_check);

	//QPushButton* apply_all_btn = new QPushButton("Apply");
	all_types_layout->addWidget(_all_types_list);
	//all_types_layout->addWidget(apply_all_btn);


	// Buttons and layout
	QPushButton* ok_btn = new QPushButton("Load");
	ok_btn->setDefault(true);
	QPushButton* cancel_btn = new QPushButton("Cancel");
	btn_layout->addWidget(ok_btn);
	btn_layout->addWidget(cancel_btn);

	// Connectors
	connect(ok_btn, SIGNAL(pressed()), this, SLOT(accept()));
	connect(cancel_btn, SIGNAL(pressed()), this, SLOT(reject()));
	connect(_all_types_list, SIGNAL(itemSelectionChanged()), this, SLOT(apply_all()));
	connect(_all_types_check, SIGNAL(stateChanged(int)), this, SLOT(all_types_check_Slot(int)));

	// Set the layouts
	main_layout->addWidget(new QLabel("Multiple possible types have been detected for these files.\nPlease select which type(s) they should belong to :"));
	main_layout->addWidget(_files_types_view);
	main_layout->addLayout(all_types_layout);
	main_layout->addLayout(btn_layout);

	setLayout(main_layout);
}

void PVInspector::PVFilesTypesSelWidget::apply_all()
{
	QList<QListWidgetItem*> selitems = _all_types_list->selectedItems();
	QList<QListWidgetItem*>::const_iterator it;
	QStringList types;
	for (it = selitems.begin(); it != selitems.end(); it++) {
		types << (*it)->text();
	}

	// Set data with model
	map_files_types::iterator it_ft;
	for (it_ft = _files_types.begin(); it_ft != _files_types.end(); it_ft++) {
		(*it_ft).second = types;
	}
	_files_types_model->emitAllTypesChanged();
}

void PVInspector::PVFilesTypesSelWidget::all_types_check_Slot(int state)
{
	bool checked = (state == Qt::Checked);
	_all_types_list->setEnabled(checked);
	_files_types_view->setEnabled(!checked);
}
