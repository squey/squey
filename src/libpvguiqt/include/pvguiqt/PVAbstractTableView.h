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

#ifndef PVGUIQT_PVABSTRACTTABLEVIEW_H
#define PVGUIQT_PVABSTRACTTABLEVIEW_H

#include <pvguiqt/PVTableView.h>
#include <pvbase/types.h>

#include <QStyledItemDelegate>
#include <QTextDocument>

namespace PVGuiQt
{

class PVAbstractTableModel;

class PVHyperlinkDelegate : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;

	void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override;

	void set_mouse_pos(const QPoint& mouse_pos_in_viewport);

	bool mouse_over_link(const QModelIndex& index, const QRect& rect) const;

private:
	bool is_url(const QModelIndex& index) const;

	void format_text_document(QTextDocument& doc, QColor& color, const QString& url, const QRect& rect, bool is_selected = false) const;

	QString get_elided_text(const QString& url, const QTextDocument& doc, int max_width) const;

private:
	QPoint _mouse_pos;
	mutable bool _mouse_over_link = false;
	const QString _link = "<a style=\"text-decoration: %4; color: %3\" href=\"%1\">%2</a>";
};

/**
 * Abstract class for Huge table view.
 *
 * It has to be used with the PVAbstractTableModel to handle
 * huge tables.
 */
class PVAbstractTableView : public PVTableView
{
	Q_OBJECT;

  public:
	explicit PVAbstractTableView(QWidget* parent = nullptr);

	/**
	 * Define the current model and update pagination information depending
	 * on its number of elements.
	 */
	void setModel(QAbstractItemModel* model) override;

	/**
	 * Get the associate model.
	 */
	PVAbstractTableModel* table_model();

	void set_hyperlink_delegate_max_index(int max_index);

  protected:
	/**
	 * Slots called on slider movement.
	 */
	void slider_move_to(int value);

	/**
	 * Clip the listing on top or bottom depending on slider position.
	 */
	void clip_slider();

	/**
	 * Update pagination when number of step in the scrollbar change.
	 */
	void new_range(int min, int max);
	void new_range();

	/**
	 * Handle action from click on the scrollbar.
	 *
	 * It handles button click but also others actions from right click.
	 */
	void scrollclick(int action);

	/**
	 * Handle selection on click and move table on last row click.
	 *
	 * Works with Shift and Control modifier for selection.
	 */
	void mousePressEvent(QMouseEvent* event) override;

	/**
	 * Handle selection and table movement
	 */
	void keyPressEvent(QKeyEvent* event) override;

	/**
	 * Move in the listing table.
	 */
	void wheelEvent(QWheelEvent* e) override;

	/**
	 * Commit in the current selection if a selection is in progress.
	 */
	void mouseReleaseEvent(QMouseEvent* event) override;

	/**
	 * Move the table in the mouse direction. It also update the
	 * "in progress" selection
	 *
	 * @note Called only when button is pressed.
	 */
	void mouseMoveEvent(QMouseEvent* event) override;

	/**
	 * Handle URL clicks
	 */
	bool viewportEvent(QEvent* event) override;

	/**
	 * Move the pagination information to have row as first line and update view.
	 *
	 * @param[in] row : row from nraw to display
	 */
	void move_to_nraw(PVRow row);

  private:
	/**
	 * Move the pagination information and update view.
	 *
	 * @param[in] row : Number of line to move by
	 */
	void move_by(int row);

	/**
	 * Move the pagination information to have to row as first line and update view.
	 *
	 * @param[in] row : row from view to display
	 */
	void move_to_row(PVRow row);

	/**
	 * Move the pagination information to be on a given page and update view.
	 *
	 * @param[in] page : Page to move on
	 */
	void move_to_page(int page);

	/**
	 * Move the pagination information to be at the end of the listing and update view.
	 */
	void move_to_end();

	/**
	 * Update view after a pagination movement
	 */
	void update_on_move();

	/**
	 * Show right_click menu once selection is done.
	 *
	 * @note : default do nothing for TableView without context menu.
	 */
	virtual void show_ctxt_menu(QPoint const&) {}

  private Q_SLOTS:
	/**
	 * Commit the selection before any right-click action.
	 */
	void show_rclick_menu(QPoint const& p);

  Q_SIGNALS:
	void selection_commited();
	void validate_selection();

  private:
	PVHyperlinkDelegate* _hyperlink_delegate = nullptr;
	size_t _hyperlink_delegate_max_index = 0;

  protected:
};
} // namespace PVGuiQt

#endif
