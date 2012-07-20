#ifndef __AXESCOMBMODEL__H_
#define __AXESCOMBMODEL__H_

#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVObserverCallback.h>

#include <QListView>

#define SUBCLASSING_VERSION 0

class AxesCombinationListModel;

class PVViewObserver : public PVHive::PVObserver<Picviz::PVView>
{
public:
	PVViewObserver(AxesCombinationListModel* model, QListView* list) : _model(model), _list(list) {}

	void refresh();

	void about_to_be_deleted()
	{
		std::cout << "PVViewObserver::about_to_be_deleted (" << boost::this_thread::get_id() << ")" << std::endl;
		_list->setEnabled(false);
		_list->reset();
	}
private:
	AxesCombinationListModel* _model;
	QListView* _list;
};

class AxesCombinationListModel : public QAbstractListModel
{
	Q_OBJECT;

	friend class PVViewObserver;

public:
	typedef PVCore::pv_shared_ptr<Picviz::PVView> PVView_p;

	AxesCombinationListModel(PVView_p& view_p, QListView* list, QObject* parent = 0) :
		QAbstractListModel(parent),
		_view_p(view_p),
		_list(list),
		_view_deleted(false)
	{

#if SUBCLASSING_VERSION
		_observer = new PVViewObserver(this, _list); // subclassing version
#else
		_observer = new PVHive::PVObserverSignal<Picviz::PVView>(this); // signal version
		_observer->connect_about_to_be_deleted(this, SLOT(about_to_be_deleted_slot(PVHive::PVObserverBase*)));
		_observer->connect_refresh(this, SLOT(refresh_slot(PVHive::PVObserverBase*)));
#endif
		PVHive::PVHive::get().register_observer(
			_view_p,
			*_observer
		);
	}

	~AxesCombinationListModel()
	{
		delete _observer;
	}

	int rowCount(const QModelIndex &parent) const
	{
		if (parent.isValid())
			return 0;

		return rowCount();
	}

	int rowCount() const
	{
		if (!_view_deleted) { // This check is needed to avoid the model crashing when deleting the view.
			return _view_p->get_axes_count();
		}
		return 0;
	}

	QVariant data(const QModelIndex &index, int role) const
	{
		if (index.row() < 0 || index.row() >= rowCount())
			return QVariant();

		QString axis_name = _view_p->get_axis_name(index.row());
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

private slots:
	void about_to_be_deleted_slot(PVHive::PVObserverBase*)
	{
		std::cout << "AxesCombinationListModel::about_to_be_deleted (" << boost::this_thread::get_id() << ")" << std::endl;
		_list->setEnabled(false);
		_list->setModel(nullptr);
		_view_deleted = true;
	}

	void refresh_slot(PVHive::PVObserverBase*)
	{
		PVLOG_INFO("AxesCombinationListModel::refresh\n");
		reset();
	}
private:
	bool _view_deleted;
	PVView_p& _view_p;
	QListView* _list;

#if SUBCLASSING_VERSION
		PVViewObserver* _observer; // subclassing version
#else
		PVHive::PVObserverSignal<Picviz::PVView>* _observer; // signal version
#endif
};

#endif // __AXESCOMBMODEL__H_
