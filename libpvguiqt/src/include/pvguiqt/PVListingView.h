/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVLISTINGVIEW_H
#define PVLISTINGVIEW_H

#include <QMenu>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/widgets/PVHelpWidget.h>

#include <pvhive/PVActor.h>
#include <pvhive/PVObserverSignal.h>

#include <inendi/PVView_types.h>

#include <pvguiqt/PVListingModel.h>
#include <pvguiqt/PVAbstractTableView.h>

#include <QHeaderView>

class QMouseEvent;

namespace PVWidgets
{

class PVHelpWidget;
}

namespace PVGuiQt
{

class PVLayerFilterProcessWidget;
class PVListingModel;

/**
 * \class PVListingView
 */
class PVListingView : public PVAbstractTableView
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
	PVListingView(Inendi::PVView_sp& view, QWidget* parent = nullptr);

	/**
	 * Clean up plugin in progress
	 */
	~PVListingView();

	/**
	 * Get associate model
	 */
	PVListingModel* listing_model();

  public slots:
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
	void enterEvent(QEvent* event) override;
	void leaveEvent(QEvent* event) override;

	/**
	 * Add nice border on hovered column
	 */
	void paintEvent(QPaintEvent* event) override;

signals:
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

  private:
	/// Getters
	Inendi::PVView const& lib_view() const { return *_obs.get_object(); }
	Inendi::PVView& lib_view() { return *_obs.get_object(); }
	PVWidgets::PVHelpWidget* help_widget() { return &_help_widget; }

  private slots:
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
	 * Highlight the column specified from an external (Hive) source.
	 *
	 * @param[in] o : Observer signal containing column information.
	 */
	void highlight_column(PVHive::PVObserverBase* o);

	/**
	 * Highlight the specified column.
	 *
	 * @param[in] col : column to highlight
	 */
	void highlight_column(int col);

	/**
	 * Notify Hive views about hovered horizontal header column.
	 *
	 * @param[in] col : Hovered column.
	 * @param[in] enter : Whether the hover begin or end.
	 */
	void section_hovered_enter(int col, bool enter);

  private:
	// Context menu
	QMenu _ctxt_menu;        //!< Context menu for right click on table cells
	QAction* _act_copy;      //!< Copy cell content action for context menu
	QAction* _act_set_color; //!< Set a color for clicked row action for context menu

	// Header context menu
	QMenu _hhead_ctxt_menu;      //!< Context menu for right click on horizontal header
	QMenu* _menu_col_count_by;   //!< Count by action for horizontal context menu
	QMenu* _menu_col_sum_by;     //!< Sum by action for horizontal context menu
	QMenu* _menu_col_min_by;     //!< Min by action for horizontal context menu
	QMenu* _menu_col_max_by;     //!< Max by action for horizontal context menu
	QMenu* _menu_col_avg_by;     //!< Average by action for horizontal context menu
	QAction* _action_col_sort;   //!< Sort a column action for horizontal context menu
	QAction* _action_col_unique; //!< Count distinct values action for horizontal context menu

	// Vertical context menu
	QMenu _vhead_ctxt_menu;          //!< Context menu for right click on vertival header
	QAction* _action_copy_row_value; //!< Copy clicked row action for vertical header action

	// Help menu
	PVWidgets::PVHelpWidget _help_widget; //!< Help menu for listing view

	// FIXME : This should be in a "context menu" context
	PVRow _ctxt_row;                                    //!< Clicked row for context menu actions
	PVCol _ctxt_col;                                    //!< Clicked col for context menu actions
	QString _ctxt_v;                                    //!< Clicked value for context menu actions
	PVGuiQt::PVLayerFilterProcessWidget* _ctxt_process; //!< Current open LayerFilter plugins widget
	std::vector<uint32_t> _headers_width;               //!< Width for each header

	int _hovered_axis = -1; //!< Hovered axis flags for paintEvent
	int _vhead_max_width;   //!< Max width for the vertical header

	// Plugins call capture local variable reference without copy making it
	// invalide at the end of the scope...
	PVCore::PVArgumentList _ctxt_args; //!< FIXME : awfull hidden global variable

  private:
	// Observers
	PVHive::PVObserverSignal<Inendi::PVView>
	    _obs; //!< Observer for current view to delete listing on view deletion
	// FIXME : It should be a PVCol instead of int
	PVHive::PVObserverSignal<int> _axis_hover_obs; //!< Observer for hovered column

	// Actor
	PVHive::PVActor<Inendi::PVView>
	    _actor; //!< Actor to emit notification about listing modification to the view
};

class PVHorizontalHeaderView : public QHeaderView
{
	Q_OBJECT

  public:
	PVHorizontalHeaderView(Qt::Orientation orientation, PVListingView* parent);

signals:
	void mouse_hovered_section(int index, bool entered);

  protected:
	bool event(QEvent* ev) override;
	void paintSection(QPainter* painter, const QRect& rect, int logicalIndex) const override;

  private:
	PVGuiQt::PVListingView* listing_view() const { return (PVGuiQt::PVListingView*)parent(); }

  private:
	int _index = -1;
};
}

#endif // PVLISTINGVIEW_H
