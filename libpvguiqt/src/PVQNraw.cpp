/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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

bool PVGuiQt::PVQNraw::show_unique_values(Inendi::PVView& view,
                                          PVRush::PVNraw const& nraw,
                                          PVCol c,
                                          Inendi::PVSelection const& sel,
                                          QWidget* parent,
                                          QDialog** dialog /*= nullptr*/)
{
	const pvcop::db::array& col_in = nraw.column(c);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;
	pvcop::db::array abs_max;

	BENCH_START(distinct_values);

	pvcop::db::array minmax;
	auto ret_pbox = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(true);

		    pvcop::db::algo::distinct(col_in, col1_out, col2_out, sel);

		    minmax = pvcop::db::algo::minmax(col2_out);

		    pvcop::db::array bit_count("number_uint64", 1);
		    auto& bc = bit_count.to_core_array<uint64_t>();
		    bc[0] = pvcop::core::algo::bit_count(sel);
		    abs_max = std::move(bit_count);
		},
	    QObject::tr("Computing values..."), parent);

	BENCH_END(distinct_values, "distinct values", col_in.size(), 4, col1_out.size(), 4);

	if (ret_pbox != PVCore::PVProgressBox::CancelState::CONTINUE ||
	    col2_out.size() == 0) { // FIXME : col1_out.size() == 0 should not happen anymore
		return false;
	}

	PVGuiQt::PVListUniqStringsDlg* dlg =
	    new PVGuiQt::PVListUniqStringsDlg(view, c, std::move(col1_out), std::move(col2_out),
	                                      std::move(abs_max), std::move(minmax), parent);
	dlg->setWindowTitle(
	    "Distinct values of axe '" +
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(c).get_name() + "'");
	dlg->show();
	if (dialog) {
		// Save the current dialog to close the old one when you open a new one.
		*dialog = dlg;
	}

	return true;
}

enum class ABS_MAX_OP { MAX, SUM, COUNT };

template <typename F>
static bool show_stats_dialog(const QString& title,
                              const F& op,
                              ABS_MAX_OP abs_max_op,
                              bool counts_are_integers,
                              Inendi::PVView& view,
                              PVRush::PVNraw const& nraw,
                              PVCol col1,
                              PVCol col2,
                              Inendi::PVSelection const& sel,
                              QWidget* parent)
{
	const pvcop::db::array& col1_in = nraw.column(col1);
	const pvcop::db::array& col2_in = nraw.column(col2);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;
	pvcop::db::array minmax;
	pvcop::db::array abs_max;

	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();

	BENCH_START(operation);

	auto ret_pbox = PVCore::PVProgressBox::progress(
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

	BENCH_END(operation, title.toStdString().c_str(), col1_in.size(), 4, col2_in.size(), 4);

	if (ret_pbox != PVCore::PVProgressBox::CancelState::CONTINUE ||
	    col1_out.size() == 0) { // FIXME : col1_out.size() == 0 should not happen anymore
		return false;
	}

	PVGuiQt::PVGroupByStringsDlg* dlg = new PVGuiQt::PVGroupByStringsDlg(
	    view, col1, col2, sel, std::move(col1_out), std::move(col2_out), std::move(abs_max),
	    std::move(minmax), counts_are_integers, parent);
	dlg->setWindowTitle(
	    title + " of axes '" +
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col1).get_name() +
	    "' and '" +
	    view.get_parent<Inendi::PVSource>().get_format().get_axes().at(col2).get_name() + "'");
	dlg->show();

	return true;
}

bool PVGuiQt::PVQNraw::show_count_by(Inendi::PVView& view,
                                     PVRush::PVNraw const& nraw,
                                     PVCol col1,
                                     PVCol col2,
                                     Inendi::PVSelection const& sel,
                                     QWidget* parent)
{
	return show_stats_dialog("Count by", &pvcop::db::algo::count_by, ABS_MAX_OP::COUNT, true, view,
	                         nraw, col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_sum_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Sum by", &pvcop::db::algo::sum_by, ABS_MAX_OP::SUM, true, view, nraw,
	                         col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_max_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Max by", &pvcop::db::algo::max_by, ABS_MAX_OP::MAX, true, view, nraw,
	                         col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_min_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Min by", &pvcop::db::algo::min_by, ABS_MAX_OP::MAX, true, view, nraw,
	                         col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_avg_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Average by", &pvcop::db::algo::average_by, ABS_MAX_OP::MAX, false,
	                         view, nraw, col1, col2, sel, parent);
}
