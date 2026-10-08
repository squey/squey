//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#ifndef PVPARALLELVIEW_PVSCATTERTHUMBNAILSVIEW_H
#define PVPARALLELVIEW_PVSCATTERTHUMBNAILSVIEW_H

#include <pvkernel/core/PVDisconnector.h>
#include <pvkernel/widgets/PVHelpWidget.h>
#include <pvkernel/widgets/PVMouseButtonsLegend.h>

#include <QModelIndex>
#include <QPixmap>
#include <QStyledItemDelegate>
#include <QWidget>

class QBoxLayout;
class QComboBox;
class QLabel;
class QLineEdit;
class QListView;
class QSlider;
class QSortFilterProxyModel;
class QTimer;

namespace Squey
{
class PVView;
} // namespace Squey

namespace PVParallelView
{

class PVScatterThumbnailsModel;

/**
 * Draws one gallery cell: the thumbnail in a fixed square, its axis pair on a
 * fixed two lines under it.
 *
 * Both halves have to be laid out here rather than left to the default item
 * delegate. That one draws the decoration at the pixmap's own size and places
 * the label right under it, so a cell whose pixmap does not match the current
 * cell size -- which is every cell while the size slider moves, since the
 * cached pixmaps still carry the size they were rendered at -- puts its label
 * somewhere else than its neighbours. Its height also follows however many
 * lines the axis names happen to wrap onto, which leaves ragged gaps between
 * rows.
 */
class PVScatterThumbnailDelegate : public QStyledItemDelegate
{
  public:
	using QStyledItemDelegate::QStyledItemDelegate;

	//! Edge of the thumbnail square, in pixels. Follows the size slider.
	void set_display_size(int size) { _display_size = size; }

	QSize sizeHint(QStyleOptionViewItem const& option, QModelIndex const& index) const override;
	void paint(QPainter* painter,
	           QStyleOptionViewItem const& option,
	           QModelIndex const& index) const override;

  private:
	int label_height(QFontMetrics const& metrics) const;

	int _display_size = 128;
};

/**
 * A gallery of every scatter view one could open on the current axes
 * combination, one thumbnail per axis pair.
 *
 * It exists to answer "which pair is worth looking at?", which until now meant
 * opening scatter views one at a time from the axis menu and guessing. Picking
 * a thumbnail opens the real PVScatterView on that pair, so the gallery is a
 * visual chooser rather than a view of its own.
 *
 * Thumbnails come from PVScatterThumbnailsModel, which renders them lazily off
 * the scaled columns. A QListView in icon mode provides the scrolling flow
 * layout, and -- because it only queries the items it paints -- the viewport
 * tracking that makes the laziness work.
 */
class PVScatterThumbnailsView : public QWidget
{
	Q_OBJECT

  public:
	explicit PVScatterThumbnailsView(Squey::PVView& view, QWidget* parent = nullptr);
	~PVScatterThumbnailsView() override;

  Q_SIGNALS:
	void set_status_bar_mouse_legend(PVWidgets::PVMouseButtonsLegend legend);
	void clear_status_bar_mouse_legend();

  protected:
	void changeEvent(QEvent* event) override;
	void enterEvent(QEnterEvent* event) override;
	void keyPressEvent(QKeyEvent* event) override;
	void leaveEvent(QEvent* event) override;

  private:
	void setup_toolbar(QBoxLayout* layout);

	//! Open the real scatter view on the pair shown by @a index.
	void open_scatter_view(QModelIndex const& index);
	void show_context_menu(QPoint const& pos);

	void schedule_preview(QModelIndex const& index);
	//! Ask the model for the enlarged render; it lands in display_preview().
	void show_preview();
	void display_preview(int row, QPixmap const& pixmap);
	void hide_preview();

	//! Re-lay out the grid at @a size, scaling the thumbnails already rendered.
	void apply_grid_size(int size);
	//! Re-render at the slider's size, once it has stopped moving.
	void apply_render_size();
	void apply_sorting(bool by_correlation);
	//! Keep only the pairs whose axis names contain @a text.
	void apply_filter(QString const& text);
	void update_counters();

	//! Row in the source model behind a (possibly sorted) view index.
	int source_row(QModelIndex const& index) const;

  private:
	Squey::PVView& _view;

	PVScatterThumbnailsModel* _model = nullptr;
	QSortFilterProxyModel* _proxy = nullptr;
	QListView* _list = nullptr;
	PVScatterThumbnailDelegate* _delegate = nullptr;

	QSlider* _size_slider = nullptr;
	/**
	 * Delays the re-render while the size slider is being dragged.
	 *
	 * Changing the render size drops every cached thumbnail, so following the
	 * slider live would throw away and redo the whole visible gallery on each
	 * step. The grid follows immediately (scaling what is already rendered);
	 * only the sharp re-render waits for the slider to settle.
	 */
	QTimer* _size_apply_timer = nullptr;
	QComboBox* _sort_combo = nullptr;
	QLineEdit* _filter_edit = nullptr;
	QLabel* _counter_label = nullptr;

	//! Free-floating enlarged render shown after the pointer settles on an item.
	QLabel* _preview = nullptr;
	QTimer* _preview_timer = nullptr;
	QModelIndex _preview_index;

	PVWidgets::PVMouseButtonsLegend _mouse_buttons_legend;
	PVWidgets::PVHelpWidget _help_widget;

	PVCore::PVDisconnector _view_deleted_connection;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERTHUMBNAILSVIEW_H
