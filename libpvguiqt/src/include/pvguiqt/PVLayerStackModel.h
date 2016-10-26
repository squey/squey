/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVLAYERSTACKMODEL_H
#define PVLAYERSTACKMODEL_H

#include <sigc++/sigc++.h>

#include <QAbstractTableModel>

#include <inendi/PVLayerStack.h>
#include <inendi/PVView.h>

#include <QBrush>
#include <QFont>

namespace Inendi
{
class PVView;
} // namespace Inendi

namespace PVGuiQt
{

/**
 * \class PVLayerStackModel
 */
class PVLayerStackModel : public QAbstractTableModel, public sigc::trackable
{
	Q_OBJECT

  public:
	explicit PVLayerStackModel(Inendi::PVView& lib_view, QObject* parent = nullptr);

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
	void show_this_layer_only(const int idx);

  public:
	Inendi::PVLayerStack const& lib_layer_stack() const { return _lib_view.get_layer_stack(); }
	Inendi::PVLayerStack& lib_layer_stack() { return _lib_view.get_layer_stack(); }
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
};
} // namespace PVGuiQt

#endif
