/**
 * \file axes-comb_model.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_AXESCOMBMODEL_H
#define PVGUIQT_AXESCOMBMODEL_H

#include <picviz/widgets/PVAD2GRFFListModel.h>
#include <picviz/PVSelRowFilteringFunction.h>
#include <picviz/PVView.h>

#include <pvkernel/core/PVSharedPointer.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVHive.h>
#include <pvhive/PVObserver.h>
#include <pvhive/PVObserverSignal.h>
#include <pvhive/PVFuncObserver.h>
#include <pvhive/PVObserverCallback.h>

#include <QListView>

#define SUBCLASSING_VERSION 0

namespace PVGuiQt {

class PVAxesCombinationModel;

namespace __impl {
class set_axis_name_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::set_axis_name), &Picviz::PVView::set_axis_name>
{
public:
	set_axis_name_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class remove_column_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::remove_column), &Picviz::PVView::remove_column>
{
public:
	remove_column_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

protected:
	virtual void about_to_be_updated(arguments_type const& args) const;
	virtual void update(arguments_type const& args) const;


private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class axis_append_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::axis_append), &Picviz::PVView::axis_append>
{
public:
	axis_append_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;
	virtual void about_to_be_updated(arguments_type const& args) const;

private:
	PVGuiQt::PVAxesCombinationModel* _model;
};

class move_axis_to_new_position_Observer: public PVHive::PVFuncObserver<Picviz::PVView, decltype(&Picviz::PVView::move_axis_to_new_position), &Picviz::PVView::move_axis_to_new_position>
{
public:
	move_axis_to_new_position_Observer(PVGuiQt::PVAxesCombinationModel* model) : _model(model) {}

protected:
	virtual void update(arguments_type const& args) const;

private:
	PVGuiQt::PVAxesCombinationModel* _model;
};


}

class PVAxesCombinationModel: public QAbstractListModel
{
	Q_OBJECT;

	friend class PVViewObserver;
	friend class __impl::set_axis_name_Observer;
	friend class __impl::remove_column_Observer;
	friend class __impl::axis_append_Observer;

public:
	PVAxesCombinationModel(Picviz::PVView_sp& view_p, QObject* parent = NULL);

public:
	int rowCount(const QModelIndex &parent) const;
	int rowCount() const;
	QVariant data(const QModelIndex &index, int role) const;
	Qt::ItemFlags flags(const QModelIndex &index) const;
	bool setData(const QModelIndex &index, const QVariant &value, int role);
	void beginInsertRow(int row);
	void endInsertRow();
	void beginRemoveRow(int row);
	void endRemoveRow();

private slots:
	void about_to_be_deleted_slot(PVHive::PVObserverBase*);
	void refresh_slot(PVHive::PVObserverBase*);
	
private:
	inline Picviz::PVView const& picviz_view() const { return *_view_observer.get_object(); }

private:
	PVHive::PVActor<Picviz::PVView> _actor;
	bool _view_deleted;

	// Observers
	__impl::set_axis_name_Observer _set_axis_name_observer; 
	PVHive::PVObserverSignal<Picviz::PVView> _view_observer;
	__impl::remove_column_Observer _remove_column_observer;
	__impl::axis_append_Observer _axis_append_observer;
	__impl::move_axis_to_new_position_Observer _move_axis_to_new_position_observer;
	PVHive::PVObserverSignal<Picviz::PVAxesCombination::columns_indexes_t> _obs_axes_comb;
};

}


#endif // __AXESCOMBMODEL__H_
