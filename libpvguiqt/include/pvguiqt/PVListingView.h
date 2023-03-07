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

#ifndef PVLISTINGVIEW_H
#define PVLISTINGVIEW_H

#include <QMenu>

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVArgument.h>
#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVAbstractTableView.h>

#include <QHeaderView>

class QMouseEvent;

namespace PVWidgets
{

class PVFilterableMenu;
class PVHelpWidget;
} // namespace PVWidgets

namespace PVGuiQt
{

class PVLayerFilterProcessWidget;
class PVListingModel;

/**
 * \class PVListingView
 */
class PVListingView : public PVAbstractTableView, public sigc::trackable
{
	Q_OBJECT
	friend class PVStatsListingWidget;
	friend class PVHorizontalHeaderView;

  public:
	/**
	 * Create a Listing view.
	 *
	 * Design widget and default connections
	 *
	 * @param view : Global display for data informations
	 * @param parent : Parent widget
	 *
	 * @note It use a view as a parameter to register observer. Thanks to this
	 * record, every view will be updated on listing model modification.
	 */
	explicit PVListingView(Squey::PVView& view, QWidget* parent = nullptr);

	/**
	 * Clean up plugin in progress
	 */
	~PVListingView() override;

	/**
	 * Get associate model
	 */
	PVListingModel* listing_model();

  public Q_SLOTS:
	/**
	 * Inform other Hive view about column click
	 */
	void section_clicked(int col);

  protected:
	/**
	 * Handle Help and goto line.
	 */
	void keyPressEvent(QKeyEvent* event) override;

	/**
	 * Resize hovered column on control modifier.
	 */
	void wheelEvent(QWheelEvent* e) override;

	/**
	 * Set correct header size on reset
	 *
	 * Reset is called by Qt
	 */
	void reset() override;

	/**
	 * Handle focus to correctly handle mouseMoveEvent
	 */
	void enterEvent(QEnterEvent* event) override;
	void leaveEvent(QEvent* event) override;

	/**
	 * Add nice border on hovered column
	 */
	void paintEvent(QPaintEvent* event) override;

  Q_SIGNALS:
	/**
	 * Signal emited to update the Stat view (lower part of listing)
	 */
	void resized();

  private:
	/**
	 * Save the QSelection in the current PVSelection and reset the QSelection
	 */
	void extract_selection();

	/**
	 * Apply the current selection and notify others views about this update.
	 *
	 * Normal selection : replace the old one
	 * Shift selection : Add unselected line in the QSelection to the PVSelection
	 * Control selection : Remove selected line to the PVSelection
	 * Shift + control selection : Remove lines not in QSelection but keep state of the others lines
	 *
	 */
	void update_view_selection_from_listing_selection();

  private:
	/**
	 * Process action from plugins (layer filter)
	 */
	void process_ctxt_menu_action(QAction const& act);

	/**
	 * Copy right clicked value in the clipboard.
	 */
	void process_ctxt_menu_copy();

	/**
	 * Prompt user for a color and apply it to selection.
	 */
	void process_ctxt_menu_set_color();

	/**
	 * Move listing to line (asked from prompt)
	 */
	void goto_line();

	/**
	 * Ask to sort the widgets based on a column value.
	 *
	 * It shows progress box and ask the model to be sorted.
	 *
	 * @param[in] col : Column to use for sort. It is column without axis combination.
	 * @param[in] order : Order to use for sort
	 */
	void sort(int col, Qt::SortOrder order);

	/**
	 * Set the given column visible in listing
	 * Used when clicking on an axis to show the corresponding column in listing
	 *
	 * @param col the column to ensure is visible in listing
	 */
	void set_section_visible(PVCombCol col);

  private:
	/// Getters
	Squey::PVView const& lib_view() const { return _view; }
	Squey::PVView& lib_view() { return _view; }
	PVWidgets::PVHelpWidget* help_widget() { return &_help_widget; }

  private Q_SLOTS:
	/**
	 * Selected the current selection (which is the current line after the
	 * first click)
	 */
	void slotDoubleClickOnVHead(int);
	void slotDoubleClickOnVHead(QModelIndex const&);

	/**
	 * Show context menu and process its actions
	 */
	void show_ctxt_menu(const QPoint& pos) override;

	/**
	 * Show horizontal header context menu and process its actions
	 */
	void show_hhead_ctxt_menu(const QPoint& pos);

	/**
	 * Show vertical header context menu and process its actions
	 */
	void show_vhead_ctxt_menu(const QPoint& pos);

	/**
	 * Set the selected color for all selected lines and notify others
	 * view about the color modification.
	 *
	 * Color is set only for the current layer.
	 *
	 * @param[in] color : Selected color
	 *
	 * @note Also use selection to know where it has to be applied.
	 */
	void set_color_selected(const PVCore::PVHSVColor& color);

	/**
	 * Save resize information for later use
	 */
	void columnResized(int column, int oldWidth, int newWidth);

	/**
	 * Highlight the specified column.
	 *
	 * @param[in] col : column to highlight
	 */
	void highlight_column(int col, bool entered);

	/**
	 * Notify Hive views about hovered horizontal header column.
	 *
	 * @param[in] col : Hovered column.
	 * @param[in] enter : Whether the hover begin or end.
	 */
	void section_hovered_enter(PVCombCol col, bool enter);

  private:
	Squey::PVView& _view;

	// Context menu
	QMenu _ctxt_menu;        //!< Context menu for right click on table cells
	QAction* _act_copy;      //!< Copy cell content action for context menu
	QAction* _act_set_color; //!< Set a color for clicked row action for context menu

	// Vertical context menu
	QMenu _vhead_ctxt_menu;          //!< Context menu for right click on vertival header
	QAction* _action_copy_row_value; //!< Copy clicked row action for vertical header action

	// Help menu
	PVWidgets::PVHelpWidget _help_widget; //!< Help menu for listing view

	// FIXME : This should be in a "context menu" context
	PVRow _ctxt_row;                                    //!< Clicked row for context menu actions
	PVCombCol _ctxt_col;                                //!< Clicked col for context menu actions
	QString _ctxt_v;                                    //!< Clicked value for context menu actions
	PVGuiQt::PVLayerFilterProcessWidget* _ctxt_process; //!< Current open LayerFilter plugins widget
	std::vector<uint32_t> _headers_width;               //!< Width for each header

	PVCombCol _hovered_axis = PVCombCol(-1); //!< Hovered axis flags for paintEvent
	int _vhead_max_width;                    //!< Max width for the vertical header

	// Plugins call capture local variable reference without copy making it
	// invalide at the end of the scope...
	PVCore::PVArgumentList _ctxt_args; //!< FIXME : awfull hidden global variable
};

class PVHorizontalHeaderView : public QHeaderView
{
	Q_OBJECT

  public:
	PVHorizontalHeaderView(Qt::Orientation orientation, PVListingView* parent);

  Q_SIGNALS:
	void mouse_hovered_section(PVCombCol index, bool entered);

  protected:
	bool event(QEvent* ev) override;
	void paintSection(QPainter* painter, const QRect& rect, int logicalIndex) const override;

  private:
	PVGuiQt::PVListingView* listing_view() const { return (PVGuiQt::PVListingView*)parent(); }

  private:
	PVCombCol _index = PVCombCol(-1);
};
} // namespace PVGuiQt

#endif // PVLISTINGVIEW_H
