/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QtCore>
#include <QtWidgets>

#include <inendi/PVCorrelationEngine.h>
#include <inendi/PVRoot.h>
#include <inendi/PVSource.h>
#include <inendi/PVView.h>

#include <pvguiqt/PVListingModel.h>

/******************************************************************************
 *
 * PVInspector::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVGuiQt::PVListingModel::PVListingModel(Inendi::PVView& view, QObject* parent)
    : PVAbstractTableModel(view.get_row_count(), parent)
    , _zombie_brush(QColor(0, 0, 0))
    , _vheader_font(":/Convergence-Regular")
    , _view(view)
{
	// Update the full model if axis combination change
	view._axis_combination_updated.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVListingModel::axes_comb_changed));

	// Call update_filter on selection update
	view._update_output_selection.connect(
	    sigc::mem_fun(this, &PVGuiQt::PVListingModel::update_filter));

	// Update filter if we change layer content
	view._update_output_layer.connect(sigc::mem_fun(this, &PVGuiQt::PVListingModel::update_filter));

	// Update display of zombie lines on option toggling
	view._toggle_zombie.connect(sigc::mem_fun(this, &PVGuiQt::PVListingModel::update_filter));

	// Update display of unselected lines on option toogling
	view._toggle_unselected.connect(sigc::mem_fun(this, &PVGuiQt::PVListingModel::update_filter));

	// Set listing view on visible_selection_listing selection.
	update_filter();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid()) {
		return {};
	}

	// Axis may have been duplicated and moved, get the real one.
	const PVCol org_col = _view.get_axes_combination().get_nraw_axis((PVCombCol)index.column());
	const PVRow r = rowIndex(index);

	if (r >= _view.get_row_count()) {
		// Nothing for rows out of bound.
		return {};
	}

	switch (role) {

	// Set content and tooltip
	case Qt::DisplayRole:
	case Qt::ToolTipRole: {
		const Inendi::PVSource& src = _view.get_parent<Inendi::PVSource>();

		return QString::fromStdString(src.get_value(r, org_col));
	}

	// Set alignment
	case Qt::TextAlignmentRole: {
		return {Qt::AlignLeft | Qt::AlignVCenter};
	}

	// Set cell color
	case Qt::BackgroundRole: {

		if (is_selected(index)) {
			// Visual selected lines from current selection
			// and "in progress" selection
			return _selection_brush;
		} else if (_view.get_real_output_selection().get_line(r)) {
			// Selected elements, use output layer color
			const PVCore::PVHSVColor color = _view.get_color_in_output_layer(r);
			return QBrush(color.toQColor());
		} else if (_view.get_line_state_in_layer_stack_output_layer(r)) {
			/* The event is unselected use darker output layer color */
			const PVCore::PVHSVColor color = _view.get_color_in_output_layer(r);
			return QBrush(color.toQColor().darker(200));
		} else {
			/* The event is a ZOMBIE */
			return _zombie_brush;
		}
	}

	// Set font color
	case (Qt::ForegroundRole): {
		// Show text in white if this is a zombie event
		if (!_view.get_real_output_selection().get_line(r) &&
		    !_view.get_line_state_in_layer_stack_output_layer(r)) {
			return QBrush(Qt::white);
		}
		return QVariant();
	}

	// Set value in italic if conversion during import has failed
	case (Qt::FontRole): {
		QFont f;

		const Inendi::PVSource& src = _view.get_parent<Inendi::PVSource>();

		if (not src.is_valid(r, org_col)) {
			f.setItalic(true);
		}

		return f;
	}
	}

	return {};
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::headerData
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::headerData(int row, Qt::Orientation orientation, int role) const
{
	PVCombCol comb_col(orientation == Qt::Horizontal ? row : 0);
	PVCol col = _view.get_axes_combination().get_nraw_axis(comb_col);

	switch (role) {
	// Horizontal header contains axis labels and Vertical is line number
	case (Qt::DisplayRole):
		if (orientation == Qt::Horizontal) {
			if (comb_col >= 0) {
				return _view.get_axis_name(comb_col);
			}
		} else if (comb_col >= 0) {
			assert(orientation == Qt::Vertical && "No others possible orientations.");
			return rowIndex(row) + 1; // Start counting rows from 1 for display
		}
		break;
	// Selected lines are bold, others use class specific font
	case (Qt::FontRole):
		if (orientation == Qt::Vertical and row >= 0) {
			if (_view.get_real_output_selection().get_line(row)) {
				QFont f(_vheader_font);
				f.setBold(true);
				return f;
			}
			return _vheader_font;
		} else if (orientation == Qt::Horizontal) {
			QFont f;
			const Inendi::PVSource& src = _view.get_parent<Inendi::PVSource>();
			f.setItalic(src.has_invalid(col) & pvcop::db::INVALID_TYPE::INVALID);
			return f;
		}
		break;
	// Define header alignment
	case (Qt::TextAlignmentRole):
		if (orientation == Qt::Horizontal) {
			return (Qt::AlignLeft + Qt::AlignTop);
		} else {
			return (Qt::AlignRight + Qt::AlignVCenter);
		}
		break;
	// Define tooltip text
	case (Qt::ToolTipRole):
		if (orientation == Qt::Horizontal) {
			const Inendi::PVRoot& root = _view.get_parent<Inendi::PVRoot>();
			const Inendi::PVCorrelation* correlation = root.correlations().correlation(&_view);

			if (correlation and correlation->col1 == col) {

				const QString& orig_source =
				    QString::fromStdString(_view.get_parent<Inendi::PVSource>().get_name());
				const QString& orig_axis = _view.get_axis_name(comb_col);
				const QString& dest_source = QString::fromStdString(
				    correlation->view2->get_parent<Inendi::PVSource>().get_name());
				const QString& dest_axis =
				    correlation->view2->get_nraw_axis_name(correlation->col2);

				return QString("Active correlation :\n%1 (%2) -> %3 (%4)")
				    .arg(orig_source)
				    .arg(orig_axis)
				    .arg(dest_source)
				    .arg(dest_axis);
			}
		}
		break;
	}

	return QVariant();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::columnCount(const QModelIndex&) const
{
	return _view.get_column_count();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::flags
 *
 *****************************************************************************/
Qt::ItemFlags PVGuiQt::PVListingModel::flags(const QModelIndex& /*index*/) const
{
	return Qt::ItemIsEnabled;
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::axes_comb_changed
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::axes_comb_changed()
{
	// Inform others widgets model is reset and view have to be reloaded
	beginResetModel();
	endResetModel();
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::sort
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::sort_on_col(PVCombCol comb_col,
                                          Qt::SortOrder order,
                                          tbb::task_group_context& ctxt)
{
	PVCol orig_col = _view.get_axes_combination().get_nraw_axis(comb_col);
	_view.sort_indexes(orig_col, _display.sorting(), &ctxt);
	if (not ctxt.is_group_execution_cancelled()) {
		sorted(comb_col, order); // set Qt sort indicator
		update_filter();
	}
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::export_line
 *
 *****************************************************************************/
QString PVGuiQt::PVListingModel::export_line(int /*row*/) const
{
	return "Not implemented.";
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::update_filter
 *
 *****************************************************************************/
void PVGuiQt::PVListingModel::update_filter()
{
	// Reset the current selection as context change
	reset_selection();

	Inendi::PVSelection const& sel = _view.get_selection_visible_listing();

	// Inform view about future update
	Q_EMIT layoutAboutToBeChanged();

	// Push selected lines
	_display.set_filter(sel);

	// Inform view new_filter is set
	Q_EMIT layoutChanged();
}
