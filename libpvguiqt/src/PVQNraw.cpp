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
	PVCore::PVProgressBox* pbox =
	    new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);

	const pvcop::db::array col_in = nraw.collection().column(c);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	double min;
	double max;
	double count;

	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();

	BENCH_START(distinct_values);

	bool ret_pbox = PVCore::PVProgressBox::progress(
	    [&, c] {
		    pvcop::db::selection s = ((pvcop::db::selection)sel).slice(0, col_in.size());

		    pvcop::db::algo::distinct(col_in, col1_out, col2_out, s);

		    pvcop::db::array minmax = pvcop::db::algo::minmax(col2_out);
		    std::string min_str = minmax.at(0);
		    std::istringstream min_buf(min_str);
		    min_buf >> min;

		    std::string max_str = minmax.at(1);
		    std::istringstream max_buf(max_str);
		    max_buf >> max;

		    count = pvcop::core::algo::bit_count(s);
		},
	    ctxt, pbox);

	BENCH_END(distinct_values, "distinct values", col_in.size(), 4, col1_out.size(), 4);

	if (!ret_pbox ||
	    col2_out.size() == 0) { // FIXME : col1_out.size() == 0 should not happen anymore
		return false;
	}

	PVGuiQt::PVListUniqStringsDlg* dlg = new PVGuiQt::PVListUniqStringsDlg(
	    view, c, std::move(col1_out), std::move(col2_out), count, min, max, parent);
	dlg->setWindowTitle("Distinct values of axe '" +
	                    view.get_parent<Inendi::PVSource>()
	                        .get_extractor()
	                        .get_format()
	                        .get_axes()
	                        .at(c)
	                        .get_name() +
	                    "'");
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
                              Inendi::PVView& view,
                              PVRush::PVNraw const& nraw,
                              PVCol col1,
                              PVCol col2,
                              Inendi::PVSelection const& sel,
                              QWidget* parent)
{
	PVCore::PVProgressBox* pbox =
	    new PVCore::PVProgressBox(QObject::tr("Computing values..."), parent);
	pbox->set_enable_cancel(true);

	const pvcop::db::array col1_in = nraw.collection().column(col1);
	const pvcop::db::array col2_in = nraw.collection().column(col2);

	pvcop::db::array col1_out;
	pvcop::db::array col2_out;

	double rel_min;
	double rel_max;
	double abs_max;

	tbb::task_group_context ctxt(tbb::task_group_context::isolated);
	ctxt.reset();

	BENCH_START(operation);

	bool ret_pbox = PVCore::PVProgressBox::progress(
	    [&, col1, col2] {
		    pvcop::db::selection s = ((pvcop::db::selection)sel).slice(0, col1_in.size());

		    op(col1_in, col2_in, col1_out, col2_out, s);

		    pvcop::db::array minmax = pvcop::db::algo::minmax(col2_out);
		    std::string min_str = minmax.at(0);
		    std::istringstream min_buf(min_str);
		    min_buf >> rel_min;

		    std::string max_str = minmax.at(1);
		    std::istringstream max_buf(max_str);
		    max_buf >> rel_max;

		    switch (abs_max_op) {
		    case ABS_MAX_OP::MAX:
			    abs_max = rel_max;
			    break;
		    case ABS_MAX_OP::SUM:
			    abs_max = pvcop::db::algo::sum(col2_out);
			    break;
		    case ABS_MAX_OP::COUNT:
			    abs_max = (double)pvcop::core::algo::bit_count(s);
			    break;
		    }
		},
	    ctxt, pbox);

	BENCH_END(operation, title.toStdString().c_str(), col1_in.size(), 4, col2_in.size(), 4);

	if (!ret_pbox ||
	    col1_out.size() == 0) { // FIXME : col1_out.size() == 0 should not happen anymore
		return false;
	}

	PVGuiQt::PVGroupByStringsDlg* dlg =
	    new PVGuiQt::PVGroupByStringsDlg(view, col1, col2, std::move(col1_out), std::move(col2_out),
	                                     abs_max, rel_min, rel_max, parent);
	dlg->setWindowTitle(title + " of axes '" +
	                    view.get_parent<Inendi::PVSource>()
	                        .get_extractor()
	                        .get_format()
	                        .get_axes()
	                        .at(col1)
	                        .get_name() +
	                    "' and '" +
	                    view.get_parent<Inendi::PVSource>()
	                        .get_extractor()
	                        .get_format()
	                        .get_axes()
	                        .at(col2)
	                        .get_name() +
	                    "'");
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
	return show_stats_dialog("Count by", &pvcop::db::algo::count_by, ABS_MAX_OP::COUNT, view, nraw,
	                         col1, col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_sum_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Sum by", &pvcop::db::algo::sum_by, ABS_MAX_OP::SUM, view, nraw, col1,
	                         col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_max_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Max by", &pvcop::db::algo::max_by, ABS_MAX_OP::MAX, view, nraw, col1,
	                         col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_min_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Min by", &pvcop::db::algo::min_by, ABS_MAX_OP::MAX, view, nraw, col1,
	                         col2, sel, parent);
}

bool PVGuiQt::PVQNraw::show_avg_by(Inendi::PVView& view,
                                   PVRush::PVNraw const& nraw,
                                   PVCol col1,
                                   PVCol col2,
                                   Inendi::PVSelection const& sel,
                                   QWidget* parent)
{
	return show_stats_dialog("Average by", &pvcop::db::algo::average_by, ABS_MAX_OP::MAX, view,
	                         nraw, col1, col2, sel, parent);
}
