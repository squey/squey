/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
