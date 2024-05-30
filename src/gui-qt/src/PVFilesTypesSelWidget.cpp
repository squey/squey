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

#include <QtCore>
#include <QtWidgets>

#include <QVBoxLayout>
#include <QDialog>
#include <QVBoxLayout>
#include <QLabel>
#include <QListWidget>

#include <PVMainWindow.h>

#include <PVFilesTypesSelWidget.h>

// Model
App::PVFilesTypesSelModel::PVFilesTypesSelModel(map_files_types& files_types,
                                                        QObject* parent)
    : QAbstractTableModel(parent), _files_types(files_types)
{
	_header_name[0] = "File path";
	_header_name[1] = "File types";
	_org_files_types = files_types;
}

int App::PVFilesTypesSelModel::rowCount(const QModelIndex& parent) const
{
	// Cf. QAbstractTableModel's documentation. This is for a table view.
	if (parent.isValid())
		return 0;

	return _files_types.size();
}

int App::PVFilesTypesSelModel::columnCount(const QModelIndex& parent) const
{
	// Same as above
	if (parent.isValid())
		return 0;

	return 2;
}

QVariant App::PVFilesTypesSelModel::data(const QModelIndex& index, int role) const
{
	if (role != Qt::DisplayRole && role != Qt::EditRole)
		return {};
	auto it = _files_types.cbegin();
	std::advance(it, index.row());
	if (index.column() == 0)
		return (*it).first;

	if (role == Qt::DisplayRole)
		return (*it).second.join("\n");

	// Provide a list with two QStringList: the first one correspond to the
	// selected items, and the second one to the original ones
	QList<QVariant> ret;
	ret << (*it).second;

	it = _org_files_types.begin();
	std::advance(it, index.row());
	ret << (*it).second;

	return ret;
}

bool App::PVFilesTypesSelModel::setData(const QModelIndex& index,
                                                const QVariant& value,
                                                int role)
{
	if (index.column() != 1 || role != Qt::EditRole)
		return false; // File name are not editable !

	auto it = _files_types.begin();
	std::advance(it, index.row());
	if (it == _files_types.end())
		return false; // Should never happen !

	(*it).second = value.toStringList();

	Q_EMIT dataChanged(index, index);

	return true;
}

Qt::ItemFlags App::PVFilesTypesSelModel::flags(const QModelIndex& index) const
{
	Qt::ItemFlags ret = Qt::ItemIsEnabled;
	if (index.column() == 1)
		ret |= Qt::ItemIsEditable;
	return ret;
}

QVariant App::PVFilesTypesSelModel::headerData(int section,
                                                       Qt::Orientation orientation,
                                                       int role) const
{
	if (orientation != Qt::Horizontal || section >= 2 || role != Qt::DisplayRole)
		return QAbstractTableModel::headerData(section, orientation, role);
	return _header_name[section];
}

void App::PVFilesTypesSelModel::emitAllTypesChanged()
{
	Q_EMIT dataChanged(QAbstractTableModel::index(1, 0),
	                   QAbstractTableModel::index(1, _files_types.size() - 1));
}

// Delegate
// Show a combo box on the second column

App::PVFilesTypesSelDelegate::PVFilesTypesSelDelegate(QObject* parent)
    : QStyledItemDelegate(parent)
{
}

QWidget* App::PVFilesTypesSelDelegate::createEditor(QWidget* parent,
                                                            const QStyleOptionViewItem& /*option*/,
                                                            const QModelIndex& /*index*/) const
{
	auto* editor = new QListWidget(parent);
	editor->setSelectionMode(QAbstractItemView::ExtendedSelection);
	return editor;
}

void App::PVFilesTypesSelDelegate::setEditorData(QWidget* editor,
                                                         const QModelIndex& index) const
{
	QList<QVariant> model_data = index.model()->data(index, Qt::EditRole).toList();
	QStringList sel_list = model_data[0].toStringList();
	QStringList org_list = model_data[1].toStringList();

	auto* listBox = static_cast<QListWidget*>(editor);
	listBox->clear();

	// Insert items
	for (auto & f : org_list) {
			bool issel = sel_list.contains(f);
		auto item = new QListWidgetItem(f, listBox);
		listBox->insertItem(-1, item);
		item->setSelected(issel);
	}
}

void App::PVFilesTypesSelDelegate::setModelData(QWidget* editor,
                                                        QAbstractItemModel* model,
                                                        const QModelIndex& index) const
{
	QStringList type_list;

	auto* listBox = static_cast<QListWidget*>(editor);
	// Get the types selected
	for (int i = 0; i < listBox->count(); i++) {
		QListWidgetItem* item = listBox->item(i);
		if (item->isSelected()) {
			type_list << item->text();
		}
	}
	model->setData(index, type_list, Qt::EditRole);
}

void App::PVFilesTypesSelDelegate::updateEditorGeometry(QWidget* editor,
                                                                const QStyleOptionViewItem& option,
                                                                const QModelIndex& /*index*/) const
{
	editor->setGeometry(option.rect);
}

QSize App::PVFilesTypesSelDelegate::sizeHint(const QStyleOptionViewItem& option,
                                                     const QModelIndex& index) const
{
	QSize ret = QStyledItemDelegate::sizeHint(option, index);
	ret.setWidth(ret.width() * 1.2);
	ret.setHeight(ret.height() * 1.2);
	return ret;
}

// Actual widget

App::PVFilesTypesSelWidget::PVFilesTypesSelWidget(App::PVMainWindow* parent,
                                                          map_files_types& files_types)
    : QDialog(parent), _files_types(files_types)
{
	main_window = parent;

	// Initalise layouts
	auto* main_layout = new QVBoxLayout();
	auto* all_types_layout = new QVBoxLayout();
	auto* btn_layout = new QHBoxLayout();

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
	_all_types_list->setSizePolicy(
	    QSizePolicy(QSizePolicy::Maximum, QSizePolicy::MinimumExpanding));
	_all_types_list->setMaximumHeight(100);

	// Compute the union and intersection of types
	QSet<QString> types_union;
	QSet<QString> types_intersec = QSet<QString>(files_types.begin()->second.begin(), files_types.begin()->second.end());

	for (auto const& file_types : files_types) {
		QSet<QString> types = QSet<QString>(file_types.second.begin(), file_types.second.end());

		types_union += types;
		types_intersec &= types;
	}

	// Set the intersection as the default selection
	// and set a gray background for the union

	for (QString const& typen : types_union) {
		auto item = new QListWidgetItem(typen, _all_types_list);
		_all_types_list->insertItem(-1, item);
		if (types_intersec.contains(typen)) {
			item->setSelected(true);
		} else {
			item->setBackground(QBrush(Qt::lightGray));
		}
	}
	_all_types_list->setEnabled(false);

	all_types_layout->addWidget(_all_types_check);

	// QPushButton* apply_all_btn = new QPushButton("Apply");
	all_types_layout->addWidget(_all_types_list);
	// all_types_layout->addWidget(apply_all_btn);

	// Buttons and layout
	auto* ok_btn = new QPushButton("Load");
	ok_btn->setDefault(true);
	auto* cancel_btn = new QPushButton("Cancel");
	btn_layout->addWidget(ok_btn);
	btn_layout->addWidget(cancel_btn);

	// Connectors
	connect(ok_btn, &QAbstractButton::pressed, this, &QDialog::accept);
	connect(cancel_btn, &QAbstractButton::pressed, this, &QDialog::reject);
	connect(_all_types_list, &QListWidget::itemSelectionChanged, this,
	        &PVFilesTypesSelWidget::apply_all);
	connect(_all_types_check, &QCheckBox::stateChanged, this,
	        &PVFilesTypesSelWidget::all_types_check_Slot);

	// Set the layouts
	main_layout->addWidget(new QLabel("Multiple possible types have been "
	                                  "detected for these files.\nPlease select "
	                                  "which type(s) they should belong to :"));
	main_layout->addWidget(_files_types_view);
	main_layout->addLayout(all_types_layout);
	main_layout->addLayout(btn_layout);

	setLayout(main_layout);
}

void App::PVFilesTypesSelWidget::apply_all()
{
	QStringList types;
	for (QListWidgetItem* elt : _all_types_list->selectedItems()) {
		types << elt->text();
	}

	// Set data with model
	for (auto& file_types : _files_types) {
		file_types.second = types;
	}
	_files_types_model->emitAllTypesChanged();
}

void App::PVFilesTypesSelWidget::all_types_check_Slot(int state)
{
	bool checked = (state == Qt::Checked);
	_all_types_list->setEnabled(checked);
	_files_types_view->setEnabled(!checked);
}
