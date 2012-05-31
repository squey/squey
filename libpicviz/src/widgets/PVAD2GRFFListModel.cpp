#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>

PVWidgets::PVAD2GRFFListModel::PVAD2GRFFListModel(const Picviz::PVView& src_view, const Picviz::PVView& dst_view, Picviz::PVTFViewRowFiltering::list_rff_t &rffs, QObject *parent /*= 0*/) :
	QAbstractTableModel(parent),
	_rffs(rffs),
	_src_view(src_view),
	_dst_view(dst_view)
{
}

int PVWidgets::PVAD2GRFFListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return _rffs.count();
}

int PVWidgets::PVAD2GRFFListModel::columnCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return 2;
}

QVariant PVWidgets::PVAD2GRFFListModel::data(const QModelIndex &index, int role) const
{
	if (index.row() < 0 || index.row() >= _rffs.size()) {
		return QVariant();
	}
	if (index.column() < 0 || index.column() >= 2) {
		return QVariant();
	}

	Picviz::PVSelRowFilteringFunction_p row_filter = _rffs.at(index.row());

	if (row_filter.get() == 0) {
		return QVariant();
	}

	if (index.column() == 1) {
		// RFF behavior
		if (role == Qt::DisplayRole)
			return QVariant(row_filter.get()->get_human_name_with_args(_src_view, _dst_view));
		if (role == Qt::UserRole) {
			QVariant ret;
			ret.setValue<void*>(row_filter.get());
			return ret;
		}
	} else {
		// binary operation behavior
		if (role == Qt::DisplayRole) {
			if (index.row() != 0) {
				return PVCore::get_binary_operation_name(row_filter.get()->get_combination_op());
			}
		} else if ((role == Qt::UserRole) || (role == Qt::EditRole)) {
			return QVariant((int)row_filter.get()->get_combination_op());
		}
	}

	return QVariant();
}

Qt::ItemFlags PVWidgets::PVAD2GRFFListModel::flags(const QModelIndex &index) const
{
	if (!index.isValid()) {
		return QAbstractItemModel::flags(index) | Qt::ItemIsDropEnabled;
	}

	Qt::ItemFlags all_flags = QAbstractItemModel::flags(index);

	if (index.column() == 1) {
		// RFF are drag & droppable
		all_flags |= Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled;
	} else {
		// the first binary operation is not editable
		if (index.row() != 0) {
			all_flags = Qt::ItemIsEditable | Qt::ItemIsEnabled;
		} else {
			all_flags &= ~(Qt::ItemIsEditable | Qt::ItemIsEnabled);
		}
	}

	return all_flags;
}

bool PVWidgets::PVAD2GRFFListModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
	if (index.row() < 0 || index.row() >= _rffs.size()) {
		return false;
	}
	if (index.column() < 0 || index.column() >= 2) {
		return false;
	}

	int col = index.column();

	if (col == 1) {
		// RFF behavior
		if (role == Qt::UserRole) {
			_rffs.replace(index.row(), ((Picviz::PVSelRowFilteringFunction*)value.value<void*>())->shared_from_this());
			emit dataChanged(index, index);
			return true;
		}
	} else {
		// binary operations behavior
		if ((role == Qt::UserRole) || (role == Qt::EditRole)) {
			_rffs.at(index.row()).get()->set_combination_op((PVCore::PVBinaryOperation)value.toInt());
			return true;
		}
	}

	return false;
}

void PVWidgets::PVAD2GRFFListModel::addRow(QModelIndex model_index, Picviz::PVSelRowFilteringFunction_p rff)
{
	int row = 0;
	if (model_index.isValid()) {
		row = model_index.row();
	}
	insertRow(row);
	if (!model_index.isValid()) {
		model_index = index(0, 1);
	}
	QVariant var;
	var.setValue<void*>(rff.get());
	setData(model_index, var, Qt::UserRole);
}

bool PVWidgets::PVAD2GRFFListModel::insertRows(int row, int count, const QModelIndex &parent)
{
    if (count < 1 || row < 0 || row > rowCount(parent))
        return false;

    beginInsertRows(QModelIndex(), row, row + count - 1);

    for (int r = 0; r < count; ++r)
        _rffs.insert(row, Picviz::PVSelRowFilteringFunction_p());

    endInsertRows();

    return true;
}

bool PVWidgets::PVAD2GRFFListModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
        return false;

    beginRemoveRows(QModelIndex(), row, row + count - 1);

    for (int r = 0; r < count; ++r)
        _rffs.removeAt(row);

    endRemoveRows();

    return true;
}

Picviz::PVTFViewRowFiltering::list_rff_t& PVWidgets::PVAD2GRFFListModel::get_rffs() const
{
    return _rffs;
}

void PVWidgets::PVAD2GRFFListModel::move_index(QModelIndex idx, bool up)
{
	QVariant var = data(idx, Qt::UserRole);

	// the following line is REQUIRED to avoid a segfault when doing the
	// following step:
	// 1- click on "Add"
	// 2- click on "Up" or "Down"
	// In that case, remove_Rows() frees the a RFF (stored as a shared
	// pointer) because it is its last reference. setData()
	// (shared_from_this more precisely) throws also the following
	// exception:
	// terminate called after throwing an instance of 'boost::exception_detail::clone_impl<boost::exception_detail::error_info_injector<boost::bad_weak_ptr> >'
	//
	// In the other cases, the PVAD2GFunctionPropertiesWidget is shown,
	// it has also a reference on it, so that the RFF is not freed.
	Picviz::PVSelRowFilteringFunction_p srff_p = _rffs.at(idx.row());

	int old_row = idx.row();
	int new_row = old_row + (up ? -1 : 1);
	removeRows(old_row, 1);
	insertRows(new_row, 1, QModelIndex());

	QModelIndex new_idx = index(new_row, idx.column(), QModelIndex());
	setData(new_idx, var, Qt::UserRole);

	//emit dataChanged(new_idx, new_idx);
}

/*Qt::DropActions PVWidgets::PVAD2GRFFListModel::supportedDropActions() const
{
    return QAbstractItemModel::supportedDropActions() | Qt::MoveAction;
}*/


/*
static bool PVWidget::ascendingLessThan(const QPair<QString, int> &s1, const QPair<QString, int> &s2)
{
    return s1.first < s2.first;
}

static bool PVWidget::decendingLessThan(const QPair<QString, int> &s1, const QPair<QString, int> &s2)
{
    return s1.first > s2.first;
}

void PVWidget::PVAD2GRFFListModel::sort(int, Qt::SortOrder order)
{
    emit layoutAboutToBeChanged();

    QList<QPair<QString, int> > list;
    for (int i = 0; i < lst.count(); ++i)
        list.append(QPair<QString, int>(lst.at(i), i));

    if (order == Qt::AscendingOrder)
        qSort(list.begin(), list.end(), ascendingLessThan);
    else
        qSort(list.begin(), list.end(), decendingLessThan);

    lst.clear();
    QVector<int> forwarding(list.count());
    for (int i = 0; i < list.count(); ++i) {
        lst.append(list.at(i).first);
        forwarding[list.at(i).second] = i;
    }

    QModelIndexList oldList = persistentIndexList();
    QModelIndexList newList;
    for (int i = 0; i < oldList.count(); ++i)
        newList.append(index(forwarding.at(oldList.at(i).row()), 0));
    changePersistentIndexList(oldList, newList);

    emit layoutChanged();
}*/
