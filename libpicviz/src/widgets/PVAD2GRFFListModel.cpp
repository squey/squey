#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>

Picviz::PVAD2GRFFListModel::PVAD2GRFFListModel(QObject *parent /*= 0*/)
    : QAbstractListModel(parent),
    _src_view(NULL),
    _dst_view(NULL)
{
}

Picviz::PVAD2GRFFListModel::PVAD2GRFFListModel(const PVView& src_view, const PVView& dst_view, const PVTFViewRowFiltering::list_rff_t &rffs, QObject *parent /*= 0*/) :
	QAbstractListModel(parent),
	lst(rffs),
	_src_view(&src_view),
	_dst_view(&dst_view)
{
}

int Picviz::PVAD2GRFFListModel::rowCount(const QModelIndex &parent) const
{
    if (parent.isValid())
        return 0;

    return lst.count();
}

QVariant Picviz::PVAD2GRFFListModel::data(const QModelIndex &index, int role) const
{
    if (index.row() < 0 || index.row() >= lst.size())
        return QVariant();

    if (role == Qt::DisplayRole)
    	return QVariant(lst.at(index.row())->get_human_name_with_args(*_src_view, *_dst_view));
    if (role == Qt::UserRole) {
    	QVariant ret;
    	ret.setValue<void*>(lst.at(index.row()).get());
    }

    return QVariant();
}

Qt::ItemFlags Picviz::PVAD2GRFFListModel::flags(const QModelIndex &index) const
{
    if (!index.isValid())
        return QAbstractItemModel::flags(index) | Qt::ItemIsDropEnabled;

    return QAbstractItemModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled;
}

bool Picviz::PVAD2GRFFListModel::setData(const QModelIndex &index, const QVariant &value, int role)
{
    /*if (index.row() >= 0 && index.row() < lst.size()
        && (role == Qt::EditRole || role == Qt::DisplayRole)) {
        lst.replace(index.row(), value.toString());
        emit dataChanged(index, index);
        return true;
    }*/
    return false;
}

bool Picviz::PVAD2GRFFListModel::insertRows(int row, int count, const QModelIndex &parent)
{
    if (count < 1 || row < 0 || row > rowCount(parent))
        return false;

    beginInsertRows(QModelIndex(), row, row + count - 1);

    for (int r = 0; r < count; ++r)
        lst.insert(row, PVSelRowFilteringFunction_p());

    endInsertRows();

    return true;
}

bool Picviz::PVAD2GRFFListModel::removeRows(int row, int count, const QModelIndex &parent)
{
    if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
        return false;

    beginRemoveRows(QModelIndex(), row, row + count - 1);

    for (int r = 0; r < count; ++r)
        lst.removeAt(row);

    endRemoveRows();

    return true;
}

/*
static bool Picviz::ascendingLessThan(const QPair<QString, int> &s1, const QPair<QString, int> &s2)
{
    return s1.first < s2.first;
}

static bool Picviz::decendingLessThan(const QPair<QString, int> &s1, const QPair<QString, int> &s2)
{
    return s1.first > s2.first;
}

void Picviz::PVAD2GRFFListModel::sort(int, Qt::SortOrder order)
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

Picviz::PVTFViewRowFiltering::list_rff_t Picviz::PVAD2GRFFListModel::getRFFList() const
{
    return lst;
}

void Picviz::PVAD2GRFFListModel::setRFFList(const PVTFViewRowFiltering::list_rff_t &rffs)
{
    lst = rffs;
    reset();
}

Qt::DropActions Picviz::PVAD2GRFFListModel::supportedDropActions() const
{
    return QAbstractItemModel::supportedDropActions() | Qt::MoveAction;
}
