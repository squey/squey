/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2015
 */

#ifndef PVGUIQT_PVABSTRACTTABLEVIEW_H
#define PVGUIQT_PVABSTRACTTABLEVIEW_H

#include <pvguiqt/PVTableView.h>
#include <pvbase/types.h>

namespace PVGuiQt
{

class PVAbstractTableModel;

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
	PVAbstractTableView(QWidget* parent = nullptr);

	/**
	 * Define the current model and update pagination information depending
	 * on its number of elements.
	 */
	void setModel(QAbstractItemModel* model) override;

	/**
	 * Get the associate model.
	 */
	PVAbstractTableModel* table_model();

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

  private slots:
	/**
	 * Commit the selection before any right-click action.
	 */
	void show_rclick_menu(QPoint const& p);

  signals:
	void validate_selection();
};
}

#endif
