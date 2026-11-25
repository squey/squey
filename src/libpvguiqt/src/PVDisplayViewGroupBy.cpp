//
// MIT License
//
// © Florent Chapelle, 2023
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvguiqt/PVDisplayViewGroupBy.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVFilterableMenu.h>

#include <squey/PVView.h>

#include <pvguiqt/PVGroupByStringsDlg.h>

template<class... Args, std::size_t... I>
auto mcimpl(auto& params, std::index_sequence<I...>)
{
	return std::make_tuple(std::any_cast<Args>(params.at(I))...);
}

template<class... Args, class Seq = std::make_index_sequence<sizeof...(Args)>>
auto many_cast(auto& params)
{
	return mcimpl<Args...>(params, Seq{});
}

QWidget* PVDisplays::PVDisplayViewGroupBy::create_widget(Squey::PVView* view,
                                                         QWidget* parent,
                                                         Params const& params) const
{
	PVGuiQt::PVGroupByStringsDlg* dlg =
	    show_group_by(*view, col_param(view, params, 0), col_param(view, params, 1),
	                  view->get_selection_visible_listing(), parent);
	delete dlg->findChild<QWidget*>("buttonBox");
	return dlg;
}

void PVDisplays::PVDisplayViewGroupBy::add_to_axis_menu(
	QMenu& menu, PVCol axis, PVCombCol axis_comb,
	Squey::PVView* view, PVDisplaysContainer* container)
{
	auto menu_col_group_by = new PVWidgets::PVFilterableMenu(axis_menu_name());
	menu_col_group_by->setIcon(toolbar_icon());
	menu.addMenu(menu_col_group_by);

	menu_col_group_by->setEnabled(not view->get_output_layer().get_selection().is_empty());

	const QStringList axes = view->get_axes_names_list();
	QList<QAction*> group_by_actions;
	
	for (PVCombCol i(0); i < axes.size(); i++) {
		if (i != axis_comb) {
			PVCol col2 = view->get_axes_combination().get_nraw_axis(i);
			const QString& axis_type = view->get_axes_combination().get_axis(i).get_type();
			if (is_groupable_by(axis_type)) {
				QAction* action_col_group_by = new QAction(axes[i], menu_col_group_by);
				group_by_actions << action_col_group_by;
				QObject::connect(action_col_group_by, &QAction::triggered, [view, axis, col2, container, this]() {
					container->create_view_widget(*this, view, {axis, col2});
				});
			}
		}
	}
	menu_col_group_by->addActions(group_by_actions);
}


enum class ABS_MAX_OP { MAX, SUM, COUNT };

template <typename F>
static auto show_stats_dialog(const QString& op_name,
                              const F& op,
                              ABS_MAX_OP abs_max_op,
                              bool counts_are_integers,
                              Squey::PVView& view,
                              PVCol col1,
                              PVCol col2,
                              Squey::PVSelection const& sel,
                              QWidget* parent) -> PVGuiQt::PVGroupByStringsDlg*
{
	auto create_groupby_model = [=](const Squey::PVView& view_v, PVCol,
	                                Squey::PVSelection const& sel) -> PVGuiQt::PVStatsModel* {
		const PVRush::PVNraw& nraw = view_v.get_rushnraw_parent();
		const pvcop::db::array& col1_in = nraw.column(col1);
		const pvcop::db::array& col2_in = nraw.column(col2);

		pvcop::db::array col1_out;
		pvcop::db::array col2_out;
		pvcop::db::array minmax;
		pvcop::db::array abs_max;

		tbb::task_group_context ctxt(tbb::task_group_context::isolated);
		ctxt.reset();

		BENCH_START(operation);

		PVCore::PVProgressBox::progress(
		    [&](PVCore::PVProgressBox& pbox) {
			    pbox.set_enable_cancel(true);

			    op(col1_in, col2_in, col1_out, col2_out, sel);

			    minmax = pvcop::db::algo::minmax(col2_out);

			    switch (abs_max_op) {
			    case ABS_MAX_OP::MAX: {
				    pvcop::db::indexes ind(1);
				    auto& id = ind.to_core_array();
				    id[0] = 1;
				    abs_max = minmax.join(ind);
			    } break;
			    case ABS_MAX_OP::SUM:
				    abs_max = pvcop::db::algo::sum(col2_out);
				    break;
			    case ABS_MAX_OP::COUNT: {
				    pvcop::db::array bit_count("number_uint64", 1);
				    auto& bc = bit_count.to_core_array<uint64_t>();
				    bc[0] = pvcop::core::algo::bit_count(sel);
				    abs_max = std::move(bit_count);
			    } break;
			    }
			},
		    QObject::tr("Computing values..."), parent);

		BENCH_END(operation, op_name.toStdString().c_str(), col1_in.size(), 4, col2_in.size(), 4);

		QString col1_name =
		    view_v.get_parent<Squey::PVSource>().get_format().get_axes().at(col1).get_name();
		QString col2_name =
		    view_v.get_parent<Squey::PVSource>().get_format().get_axes().at(col2).get_name();

		return new PVGuiQt::PVStatsModel(op_name, col1_name, col2_name, std::move(col1_out),
		                                 std::move(col2_out), std::move(abs_max),
		                                 std::move(minmax));
	};

	PVGuiQt::PVGroupByStringsDlg* dlg = new PVGuiQt::PVGroupByStringsDlg(
	    view, col1, col2, create_groupby_model, sel, counts_are_integers, parent);
	dlg->setWindowTitle(
	    op_name + " by of axes '" +
	    view.get_parent<Squey::PVSource>().get_format().get_axes().at(col1).get_name() +
	    "' and '" +
	    view.get_parent<Squey::PVSource>().get_format().get_axes().at(col2).get_name() + "'");

	return dlg;
}

auto PVDisplays::PVDisplayViewCountBy::show_group_by(
    Squey::PVView& view, PVCol col1, PVCol col2, Squey::PVSelection const& sel, QWidget* parent) const
	-> PVGuiQt::PVGroupByStringsDlg*
{
	return show_stats_dialog("Count", &pvcop::db::algo::count_by, ABS_MAX_OP::COUNT, true, view,
	                         col1, col2, sel, parent);
}

auto PVDisplays::PVDisplayViewSumBy::show_group_by(
    Squey::PVView& view, PVCol col1, PVCol col2, Squey::PVSelection const& sel, QWidget* parent) const
	-> PVGuiQt::PVGroupByStringsDlg*
{
	const PVRush::PVNraw& nraw = view.get_rushnraw_parent();
	bool counts_are_integers = nraw.column(col2).formatter()->name() != "number_float" and
	                           nraw.column(col2).formatter()->name() != "number_double";

	QStringList signed_types = {"number_int8",  "number_int16", "number_int32",
	                            "number_int64", "number_float", "number_double"};

	ABS_MAX_OP max_op = ABS_MAX_OP::SUM;

	const QString& column_type = view.get_axes_combination().get_axis(col2).get_type();
	if (signed_types.contains(column_type)) {
		max_op = ABS_MAX_OP::MAX;
	}

	return show_stats_dialog("Sum", &pvcop::db::algo::sum_by, max_op, counts_are_integers, view,
	                         col1, col2, sel, parent);
}

auto PVDisplays::PVDisplayViewMaxBy::show_group_by(
    Squey::PVView& view, PVCol col1, PVCol col2, Squey::PVSelection const& sel, QWidget* parent) const
	-> PVGuiQt::PVGroupByStringsDlg*
{
	return show_stats_dialog("Max", &pvcop::db::algo::max_by, ABS_MAX_OP::MAX, true, view, col1,
	                         col2, sel, parent);
}

auto PVDisplays::PVDisplayViewMinBy::show_group_by(
    Squey::PVView& view, PVCol col1, PVCol col2, Squey::PVSelection const& sel, QWidget* parent) const
	-> PVGuiQt::PVGroupByStringsDlg*
{
	return show_stats_dialog("Min", &pvcop::db::algo::min_by, ABS_MAX_OP::MAX, true, view, col1,
	                         col2, sel, parent);
}

auto PVDisplays::PVDisplayViewAverageBy::show_group_by(
    Squey::PVView& view, PVCol col1, PVCol col2, Squey::PVSelection const& sel, QWidget* parent) const
	-> PVGuiQt::PVGroupByStringsDlg*
{
	return show_stats_dialog("Average", &pvcop::db::algo::average_by, ABS_MAX_OP::MAX, false, view,
	                         col1, col2, sel, parent);
}

