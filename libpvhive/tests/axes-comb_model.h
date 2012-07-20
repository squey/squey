#ifndef __AXESCOMBMODEL__H_
#define __AXESCOMBMODEL__H_

#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVObserverCallback.h>

#include <QListView>

#define SUBCLASSING_VERSION 0

class AxesCombinationListModel;

class set_axis_name_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::set_axis_name), &Picviz::PVView::set_axis_name>
{
public:
	set_axis_name_Observer(AxesCombinationListModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	AxesCombinationListModel* _model;
};

class remove_column_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::remove_column), &Picviz::PVView::remove_column>
{
public:
	remove_column_Observer(AxesCombinationListModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	AxesCombinationListModel* _model;
};

class axis_append_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::axis_append), &Picviz::PVView::axis_append>
{
public:
	axis_append_Observer(AxesCombinationListModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	AxesCombinationListModel* _model;
};

class move_axis_to_new_position_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::move_axis_to_new_position), &Picviz::PVView::move_axis_to_new_position>
{
public:
	move_axis_to_new_position_Observer(AxesCombinationListModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	AxesCombinationListModel* _model;
};

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
	friend class set_axis_name_Observer;
	friend class remove_column_Observer;
	friend class axis_append_Observer;

public:
	typedef PVCore::pv_shared_ptr<Picviz::PVView> PVView_p;

	AxesCombinationListModel(PVView_p& view_p, QListView* list, QObject* parent = 0) :
		QAbstractListModel(parent),
		_view_p(view_p),
		_list(list),
		_view_deleted(false)
	{
		// PVView observer signal
		auto view_observer = new PVHive::PVObserverSignal<Picviz::PVView>(this);
		PVHive::PVHive::get().register_observer(
			_view_p,
			*view_observer
		);
		view_observer->connect_about_to_be_deleted(this, SLOT(about_to_be_deleted_slot(PVHive::PVObserverBase*)));
		_observers.push_back(view_observer);

		// PVView::set_axis_name function observer
		set_axis_name_Observer* set_axis_name_observer = new set_axis_name_Observer(this);
		PVHive::PVHive::get().register_func_observer(
			_view_p,
			*set_axis_name_observer
		);
		_observers.push_back(set_axis_name_observer);

		// PVView::remove_column function observer
		remove_column_Observer* remove_column_observer = new remove_column_Observer(this);
		PVHive::PVHive::get().register_func_observer(
			_view_p,
			*remove_column_observer
		);
		_observers.push_back(remove_column_observer);

		// PVView::axis_append function observer
		axis_append_Observer* axis_append_observer = new axis_append_Observer(this);
		PVHive::PVHive::get().register_func_observer(
			_view_p,
			*axis_append_observer
		);
		_observers.push_back(axis_append_observer);

		// PVView::move_axis_to_new_position function observer
		move_axis_to_new_position_Observer* move_axis_to_new_position_observer = new move_axis_to_new_position_Observer(this);
		PVHive::PVHive::get().register_func_observer(
			_view_p,
			*move_axis_to_new_position_observer
		);
		_observers.push_back(move_axis_to_new_position_observer);
	}

	~AxesCombinationListModel()
	{
		for (PVHive::PVObserverObjectBase* observer : _observers) {
			delete observer;
		}
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

	bool insertRows(int row, int count, const QModelIndex &parent = QModelIndex())
	{
		if (count < 1 || row < 0 || row > rowCount())
			return false;

		beginInsertRows(QModelIndex(), row, row + count - 1);
		endInsertRows();

		return true;
	}

	bool removeRows(int row, int count, const QModelIndex &parent = QModelIndex())
	{
		if (count <= 0 || row < 0 || (row + count) > rowCount())
			return false;

		beginRemoveRows(QModelIndex(), row, row + count - 1);
		endRemoveRows();

		return true;
	}

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
	std::vector<PVHive::PVObserverObjectBase*> _observers;
	PVView_p& _view_p;
	QListView* _list;
	bool _view_deleted;
};

#endif // __AXESCOMBMODEL__H_
