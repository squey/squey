#ifndef __AXESCOMBMODEL__H_
#define __AXESCOMBMODEL__H_

#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverCallback.h>

class AxesCombinationListModel;

class ModelIndexObserver : public PVHive::PVObserver<Picviz::PVView>
{
public:
	ModelIndexObserver(const AxesCombinationListModel* parent, uint32_t row) : _parent(parent), _row(row) {}

	void refresh();

	void about_to_be_deleted()
	{
		PVLOG_INFO("about_to_be_deleted\n");
	}

private:
	const AxesCombinationListModel* _parent;
	uint32_t _row;
};

class AxesCombinationListModel : public QAbstractListModel
{
	Q_OBJECT;
	friend class ModelIndexObserver;
public:
	typedef PVCore::pv_shared_ptr<Picviz::PVView> PVView_p;

	AxesCombinationListModel(PVView_p view_p, QObject* parent = 0) :
		QAbstractListModel(parent),
		_view_p(view_p)
	{
		for (int i = 0 ; i < rowCount() ; i++) {
			insertRows(i, 1);
		}
	}

	QModelIndex index(int row, int column, const QModelIndex & parent = QModelIndex()) const
	{
		PVLOG_INFO("persistentIndexList().size()=%d\n", persistentIndexList().size());

		QPersistentModelIndex idx_pst(getPersistentIndex(row));
		if (!idx_pst.isValid()) {
			ModelIndexObserver* observer = new ModelIndexObserver(this, row);

			PVHive::PVHive::get().register_observer(
				_view_p,
				//[&](Picviz::PVView const & view) { return view.get_axis_name(row); },
				*observer
			);
			QPersistentModelIndex ret(createIndex(row, column, observer));
			_pst_idx << ret;
			return ret;
		}
		return idx_pst;
	}

	QPersistentModelIndex getPersistentIndex(int row) const
	{
		for (QPersistentModelIndex const& idx : persistentIndexList()) {
			if (idx.row() == row) {
				return idx;
			}
		}
		return QPersistentModelIndex();
	}

	int rowCount(const QModelIndex &parent) const
	{
		if (parent.isValid())
			return 0;

		return rowCount();
	}

	int rowCount() const
	{

		return _view_p->get_axes_count();
	}

	QVariant data(const QModelIndex &index, int role) const
	{
		if (index.row() < 0 || index.row() >= rowCount())
			return QVariant();

		QString axis_name = _view_p->get_axes_names_list().at(index.row());
		if (role == Qt::DisplayRole)
			return QVariant(axis_name);

		return QVariant();
	}

	Qt::ItemFlags flags(const QModelIndex &index) const
	{
		if (!index.isValid())
			return QAbstractItemModel::flags(index) | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;

		return QAbstractItemModel::flags(index) | Qt::ItemIsDragEnabled | Qt::ItemIsDropEnabled | Qt::ItemIsEditable;
	}

	bool setData(const QModelIndex &index, const QVariant &value, int role)
	{
		PVLOG_INFO("setData\n");
		if (index.row() >= 0 && index.row() < rowCount()) {
			if (role == Qt::EditRole) {
				_view_p->set_axis_name(index.row(), value.toString());
				emit dataChanged(index, index);
				return true;
			}
		}
		return false;
	}

	/*void addRow(QModelIndex model_index)
	{
		PVLOG_INFO("addRow\n");
		int row = 0;
		if (model_index.isValid()) {
			row = model_index.row();
		}
		insertRow(row);
		if (!model_index.isValid()) {
			model_index = index(0, 0);
		}
		QVariant var;
		var.setValue<void*>(rff.get());
		setData(model_index, var, Qt::UserRole);
	}*/

	/*bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex())
	{
		PVLOG_INFO("insertRows\n");
		if (count < 1 || row < 0 || row > rowCount(parent))
			return false;

		beginInsertRows(QModelIndex(), row, row + count - 1);


		endInsertRows();

		return true;
	}*/

	/*bool removeRows(int row, int count, const QModelIndex &parent)
	{
		if (count <= 0 || row < 0 || (row + count) > rowCount(parent))
			return false;

		beginRemoveRows(QModelIndex(), row, row + count - 1);

		for (int r = 0; r < count; ++r)
			_rffs.removeAt(row);

		endRemoveRows();

		return true;
	}*/

private:
	mutable PVView_p _view_p;
	mutable QList<QPersistentModelIndex> _pst_idx;
};

#endif // __AXESCOMBMODEL__H_
