/**
 * \file axes-comb_model.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVGUIQT_AXESLISTMODEL_H
#define PVGUIQT_AXESLISTMODEL_H

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

class PVAxesListModel: public QAbstractListModel
{
	Q_OBJECT;

public:
	PVAxesListModel(Picviz::PVView_sp& view_p, QObject* parent = NULL);

public:
	int rowCount(const QModelIndex &parent) const override;
	int rowCount() const;
	QVariant data(const QModelIndex &index, int role) const override;
	Qt::ItemFlags flags(const QModelIndex &index) const override;

private slots:
	void about_to_be_deleted_slot(PVHive::PVObserverBase*);
	void refresh_slot(PVHive::PVObserverBase*);
	
protected:
	inline Picviz::PVView const& picviz_view() const { return *_view_observer.get_object(); }

private:
	bool _view_deleted;

	// Observers
	PVHive::PVObserverSignal<Picviz::PVView> _view_observer;
};

}


#endif // __AXESLISTMODEL__H_
