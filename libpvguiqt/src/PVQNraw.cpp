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

#include <QDialog>

#include <algorithm> // std::max_element

#include <inendi/PVSource.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVNraw.h>

#include <pvguiqt/PVListUniqStringsDlg.h>
#include <pvguiqt/PVGroupByStringsDlg.h>
#include <pvguiqt/PVQNraw.h>

#include <pvcop/db/algo.h>

#include <pvkernel/core/inendi_bench.h>

PVGuiQt::PVStatsModel*
distinct_values_create_model(const Inendi::PVView& view, PVCol c, Inendi::PVSelection const& sel)
{
	const PVRush::PVNraw& nraw = view.get_rushnraw_parent();
	const pvcop::db::array& col_in = nraw.column(c);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;
	pvcop::db::array abs_max;

	BENCH_START(distinct_values);

	pvcop::db::array minmax;
	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(true);

		    pvcop::db::algo::distinct(col_in, col1_out, col2_out, sel);

		    minmax = pvcop::db::algo::minmax(col2_out);

		    pvcop::db::array bit_count("number_uint64", 1);
		    auto& bc = bit_count.to_core_array<uint64_t>();
		    bc[0] = pvcop::core::algo::bit_count(sel);
		    abs_max = std::move(bit_count);
		},
	    QObject::tr("Computing values..."), /*parent*/ nullptr);

	BENCH_END(distinct_values, "distinct values", col_in.size(), 4, col1_out.size(), 4);

	QString col1_name =
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(c).get_name();

	return new PVGuiQt::PVStatsModel("Count", col1_name, QString(), std::move(col1_out),
	                                 std::move(col2_out), std::move(abs_max), std::move(minmax));
}

bool PVGuiQt::PVQNraw::show_unique_values(Inendi::PVView& view,
                                          PVCol c,
                                          QWidget* parent,
                                          QDialog** dialog /*= nullptr*/)
{
	PVGuiQt::PVListUniqStringsDlg* dlg =
	    new PVGuiQt::PVListUniqStringsDlg(view, c, &distinct_values_create_model, parent);
	dlg->show();
	if (dialog) {
		// Save the current dialog to close the old one when you open a new one.
		*dialog = dlg;
	}

	return true;
}

enum class ABS_MAX_OP { MAX, SUM, COUNT };

template <typename F>
static bool show_stats_dialog(const QString& op_name,
                              const F& op,
                              ABS_MAX_OP abs_max_op,
                              bool counts_are_integers,
                              Inendi::PVView& view,
                              PVCol col1,
                              PVCol col2,
                              Inendi::PVSelection const& sel,
                              QWidget* parent)
{
	auto create_groupby_model = [=](const Inendi::PVView& view, PVCol,
	                                Inendi::PVSelection const& sel) -> PVGuiQt::PVStatsModel* {
		const PVRush::PVNraw& nraw = view.get_rushnraw_parent();
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
		    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col1).get_name();
		QString col2_name =
		    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col2).get_name();

		return new PVGuiQt::PVStatsModel(op_name, col1_name, col2_name, std::move(col1_out),
		                                 std::move(col2_out), std::move(abs_max),
		                                 std::move(minmax));
	};

	PVGuiQt::PVGroupByStringsDlg* dlg = new PVGuiQt::PVGroupByStringsDlg(
	    view, col1, col2, create_groupby_model, sel, counts_are_integers, parent);
	dlg->setWindowTitle(
	    op_name + " by of axes '" +
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col1).get_name() +
	    "' and '" +
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col2).get_name() + "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_count_by(
    Inendi::PVView& view, PVCol col1, PVCol col2, Inendi::PVSelection const& sel, QWidget* parent)
{
	return show_stats_dialog("Count", &pvcop::db::algo::count_by, ABS_MAX_OP::COUNT, true, view,
	                         col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_sum_by(
    Inendi::PVView& view, PVCol col1, PVCol col2, Inendi::PVSelection const& sel, QWidget* parent)
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

bool PVGuiQt::PVQNraw::show_max_by(
    Inendi::PVView& view, PVCol col1, PVCol col2, Inendi::PVSelection const& sel, QWidget* parent)
{
	return show_stats_dialog("Max", &pvcop::db::algo::max_by, ABS_MAX_OP::MAX, true, view, col1,
	                         col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_min_by(
    Inendi::PVView& view, PVCol col1, PVCol col2, Inendi::PVSelection const& sel, QWidget* parent)
{
	return show_stats_dialog("Min", &pvcop::db::algo::min_by, ABS_MAX_OP::MAX, true, view, col1,
	                         col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_avg_by(
    Inendi::PVView& view, PVCol col1, PVCol col2, Inendi::PVSelection const& sel, QWidget* parent)
{
	return show_stats_dialog("Average", &pvcop::db::algo::average_by, ABS_MAX_OP::MAX, false, view,
	                         col1, col2, sel, parent);
}
