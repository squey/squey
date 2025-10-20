//
// MIT License
//
// © ESI Group, 2015
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

#include <QtCore>
#include <QtWidgets>

#include <squey/PVCorrelationEngine.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>
#include <pvguiqt/PVListingModel.h>

#include <boost/algorithm/string/replace.hpp>

/******************************************************************************
 *
 * App::PVListingModel::PVListingModel
 *
 *****************************************************************************/
PVGuiQt::PVListingModel::PVListingModel(Squey::PVView& view, QObject* parent)
    : PVAbstractTableModel(view.get_row_count(), parent)
    , _zombie_brush(QColor(0, 0, 0))
    , _vheader_font(":/Convergence-Regular")
    , _view(&view)
{
	// Update the full model if axis combination change
	view._axis_combination_updated.connect(
	    sigc::mem_fun(*this, &PVGuiQt::PVListingModel::axes_comb_changed));

	// Call update_filter on selection update
	view._update_output_selection.connect([&](){
		QMetaObject::invokeMethod(qApp, [&](){
            update_filter();
        }, Qt::QueuedConnection);
	});

	// Update filter if we change layer content
	view._update_output_layer.connect(sigc::mem_fun(*this, &PVGuiQt::PVListingModel::update_filter));

	// Update display of zombie lines on option toggling
	view._toggle_zombie.connect(sigc::mem_fun(*this, &PVGuiQt::PVListingModel::update_filter));

	// Update display of unselected lines on option toogling
	view._toggle_unselected.connect(sigc::mem_fun(*this, &PVGuiQt::PVListingModel::update_filter));

	view._about_to_be_delete.connect([this](){ _view = nullptr; });

	// Set listing view on visible_selection_listing selection.
	update_filter();
}

static QBrush black_or_white_best_contrast(const QBrush& brush) // https://stackoverflow.com/a/1855903/340754
{
	const QColor& color = brush.color();

	double luminance = (0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()) / 255;

	int c = (luminance > 0.5) ? 0 : 255;

	return QBrush(QColor(c, c, c));
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::data
 *
 *****************************************************************************/
QVariant PVGuiQt::PVListingModel::data(const QModelIndex& index, int role) const
{
	if (not index.isValid() or _view == nullptr) {
		return {};
	}

	// Axis may have been duplicated and moved, get the real one.
	const PVCol org_col = _view->get_axes_combination().get_nraw_axis((PVCombCol)index.column());
	const PVRow r = rowIndex(index);

	if (r >= _view->get_row_count()) {
		// Nothing for rows out of bound.
		return {};
	}

	auto get_background_brush = [&](const QModelIndex& index) {
		if (is_selected(index)) {
			// Visual selected lines from current selection
			// and "in progress" selection
			return _selection_brush;
		} else if (_view->get_real_output_selection().get_line(r)) {
			// Selected elements, use output layer color
			const PVCore::PVHSVColor color = _view->get_color_in_output_layer(r);
			if (color == HSV_COLOR_WHITE or color == HSV_COLOR_BLACK) {
				return QBrush();
			}
			else {
				return QBrush(color.toQColor());
			}
		} else if (_view->get_line_state_in_layer_stack_output_layer(r)) {
			/* The event is unselected use darker output layer color */
			const PVCore::PVHSVColor color = _view->get_color_in_output_layer(r);
			return QBrush(color.toQColor().darker(200));
		} else {
			/* The event is a ZOMBIE */
			return _zombie_brush;
		}
	};

	switch (role) {

	// Set content and tooltip
	case Qt::DisplayRole: {
		const auto& src = _view->get_parent<Squey::PVSource>();
		return QString::fromStdString(src.get_value(r, org_col));
	}
	case Qt::ToolTipRole: {
		const auto& src = _view->get_parent<Squey::PVSource>();
		std::string str_with_newlines = src.get_value(r, org_col);
		boost::replace_all(str_with_newlines, "\\n", "<br>"); // Properly show new lines
		return get_wrapped_string(QString::fromStdString(str_with_newlines));
	}

	// Set alignment
	case Qt::TextAlignmentRole: {
		return {Qt::AlignLeft | Qt::AlignVCenter};
	}

	// Set cell color
	case Qt::BackgroundRole: {
		return get_background_brush(index);
	}

	// Set font color
	case (Qt::ForegroundRole): {
		const QBrush& bg_brush = get_background_brush(index);
		if (bg_brush == QBrush()) {
			return PVCore::PVTheme::is_color_scheme_light() ? QBrush(Qt::black) : QBrush(Qt::white);
		}
		return black_or_white_best_contrast(bg_brush);
	}

	// Set value in italic if conversion during import has failed
	case (Qt::FontRole): {
		QFont f;

		const auto& src = _view->get_parent<Squey::PVSource>();

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
	// Sometimes Qt is using an invalid row value...
	if (_view == nullptr or (role == Qt::DisplayRole and orientation == Qt::Vertical and row>= (int)_view->get_row_count())) {
		return {};
	}

	PVCombCol comb_col(orientation == Qt::Horizontal ? row : 0);
	PVCol col = _view->get_axes_combination().get_nraw_axis(comb_col);

	switch (role) {
	// Horizontal header contains axis labels and Vertical is line number
	case (Qt::DisplayRole):
		if (orientation == Qt::Horizontal) {
			if (comb_col >= 0) {
				return _view->get_axis_name(comb_col);
			}
		} else if (comb_col >= 0) {
			assert(orientation == Qt::Vertical && "No others possible orientations.");
			return rowIndex(row) + 1; // Start counting rows from 1 for display
		}
		break;
	// Selected lines are bold, others use class specific font
	case (Qt::FontRole):
		if (orientation == Qt::Vertical and row >= 0) {
			if (_view->get_real_output_selection().get_line(row)) {
				QFont f(_vheader_font);
				f.setBold(true);
				return f;
			}
			return _vheader_font;
		} else if (orientation == Qt::Horizontal) {
			QFont f;
			const auto& src = _view->get_parent<Squey::PVSource>();
			f.setItalic(src.has_invalid(col) & pvcop::db::INVALID_TYPE::INVALID);
			return f;
		}
		break;
	// Define header alignment
	case (Qt::TextAlignmentRole):
		if (orientation == Qt::Horizontal) {
			return {Qt::AlignLeft | Qt::AlignTop};
		} else {
			return {Qt::AlignRight | Qt::AlignVCenter};
		}
		break;
	// Define tooltip text
	case (Qt::ToolTipRole):
		if (orientation == Qt::Horizontal) {
			const auto& root = _view->get_parent<Squey::PVRoot>();
			const Squey::PVCorrelation* correlation = root.correlations().correlation(_view);

			if (correlation and correlation->col1 == col) {

				QString orig_source =
				    QString::fromStdString(_view->get_parent<Squey::PVSource>().get_name());
				QString orig_axis = _view->get_axis_name(comb_col);
				QString dest_source = QString::fromStdString(
				    correlation->view2->get_parent<Squey::PVSource>().get_name());
				QString dest_axis =
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

	return {};
}

/******************************************************************************
 *
 * PVGuiQt::PVListingModel::columnCount
 *
 *****************************************************************************/
int PVGuiQt::PVListingModel::columnCount(const QModelIndex&) const
{
	return _view == nullptr ? 0 : _view->get_column_count();
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
void PVGuiQt::PVListingModel::axes_comb_changed(bool /*async*/)
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
    if (_view == nullptr) {
        return;
    }

	if (comb_col == -1) { // sort "index" virtual column
		auto& indexes = _display.sorting().to_core_array();
		std::iota(indexes.begin(), indexes.end(), 0);
	}
	else {
		PVCol orig_col = _view->get_axes_combination().get_nraw_axis(comb_col);
		_view->sort_indexes(orig_col, _display.sorting(), &ctxt);
	}

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
QString PVGuiQt::PVListingModel::export_line(int /*row*/, const QString& /*fsep*/) const
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
    if (_view == nullptr) {
        return;
    }

	// Reset the current selection as context change
	reset_selection();

	Squey::PVSelection const& sel = _view->get_selection_visible_listing();

	// Inform view about future update
	Q_EMIT layoutAboutToBeChanged();

	// Push selected lines
	_display.set_filter(sel);

	// Inform view new_filter is set
	Q_EMIT layoutChanged();
}
