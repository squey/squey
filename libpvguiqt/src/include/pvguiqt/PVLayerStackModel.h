/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVLAYERSTACKMODEL_H
#define PVLAYERSTACKMODEL_H

#include <QAbstractTableModel>

#include <inendi/PVLayerStack.h>
#include <inendi/PVView_types.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <QBrush>
#include <QFont>

namespace Inendi
{
class PVView;
}

namespace PVGuiQt
{

/**
 * \class PVLayerStackModel
 */
class PVLayerStackModel : public QAbstractTableModel
{
	Q_OBJECT

  public:
	PVLayerStackModel(Inendi::PVView_sp& lib_view, QObject* parent = NULL);

  public:
	int columnCount(const QModelIndex& index) const override;
	QVariant data(const QModelIndex& index, int role) const override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	int rowCount(const QModelIndex& index = QModelIndex()) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;

  public:
	void delete_layer_n(const int idx);
	void delete_selected_layer();
	void duplicate_selected_layer(const QString& name);
	void add_new_layer(QString name);
	void move_selected_layer_up();
	void move_selected_layer_down();
	void reset_layer_colors(const int idx);

  public:
	Inendi::PVLayerStack const& lib_layer_stack() const { return _lib_view.get_layer_stack(); }
	Inendi::PVLayerStack& lib_layer_stack() { return _lib_view.get_layer_stack(); }
	PVHive::PVActor<Inendi::PVView>& view_actor() { return _actor; }
	Inendi::PVView const& lib_view() const { return _lib_view; }
	Inendi::PVView& lib_view() { return _lib_view; }

  private:
	inline int lib_index_from_model_index(int model_index) const
	{
		assert(model_index < rowCount());
		return rowCount() - model_index - 1;
	}

  private Q_SLOTS:
	void layer_stack_about_to_be_refreshed();
	void layer_stack_refreshed();

  private:
	Inendi::PVView& _lib_view;
	QBrush select_brush;   //!<
	QFont select_font;     //!<
	QBrush unselect_brush; //!<
	QFont unselect_font;   //!<

	PVHive::PVActor<Inendi::PVView> _actor;
};
}

#endif
