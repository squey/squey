/**
 * \file PVLayerStackModel.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVLAYERSTACKMODEL_H
#define PVLAYERSTACKMODEL_H

#include <QAbstractTableModel>

#include <picviz/PVLayerStack.h>
#include <picviz/PVView_types.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <QBrush>
#include <QFont>

namespace PVGuiQt {

/**
 * \class PVLayerStackModel
 */
class PVLayerStackModel : public QAbstractTableModel
{
	Q_OBJECT

public:
	PVLayerStackModel(Picviz::PVView_sp& lib_view, QObject* parent = NULL);

public:
	int columnCount(const QModelIndex &index) const override;
	QVariant data(const QModelIndex &index, int role) const override;
	Qt::ItemFlags flags(const QModelIndex &index) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	int rowCount(const QModelIndex &index = QModelIndex()) const override;
	bool setData(const QModelIndex &index, const QVariant &value, int role) override;

public:
	void delete_layer_n(const int idx);
	void delete_selected_layer();
	void add_new_layer();
	void add_new_layer_from_file(const QString& path);
	void load_from_file(const QString& file);

public:
	Picviz::PVLayerStack const& lib_layer_stack() const { return *_obs.get_object(); }
	Picviz::PVLayerStack& lib_layer_stack() { return *_obs.get_object(); }

private slots:
	void layer_stack_about_to_be_deleted(PVHive::PVObserverBase* o);
	void layer_stack_about_to_be_refreshed(PVHive::PVObserverBase* o);
	void layer_stack_refreshed(PVHive::PVObserverBase* o);

private:
	QBrush select_brush;       //!<
	QFont select_font;         //!<
	QBrush unselect_brush;     //!<
	QFont unselect_font;       //!<

	PVHive::PVActor<Picviz::PVView> _actor;
	PVHive::PVObserverSignal<Picviz::PVLayerStack> _obs;

	bool _ls_valid;
};

}

#endif
