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

#include <pvguiqt/PVDisplayViewDistinctValues.h>

#include <pvkernel/core/PVProgressBox.h>

#include <squey/PVView.h>

#include <pvguiqt/PVListUniqStringsDlg.h>

namespace PVDisplays
{
	auto distinct_values_create_model(Squey::PVView const& view, PVCol c, Squey::PVSelection const& sel) -> PVGuiQt::PVStatsModel*;
}

PVDisplays::PVDisplayViewDistinctValues::PVDisplayViewDistinctValues()
    : PVDisplayViewIf(ShowInCtxtMenu,
                      QObject::tr("Distinct values"),
                      QIcon(":/fileslist_black"),
					  QObject::tr("Distinct values"),
					  Qt::LeftDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewDistinctValues::create_widget(Squey::PVView* view,
                                                                QWidget* parent,
                                                                Params const& params) const
{
	auto* dlg = new PVGuiQt::PVListUniqStringsDlg(*view, std::any_cast<PVCol>(params.at(0)), &distinct_values_create_model, parent);
	delete dlg->findChild<QWidget*>("buttonBox");
	return dlg;
}

namespace PVDisplays
{

PVGuiQt::PVStatsModel* distinct_values_create_model(const Squey::PVView& view, PVCol c, Squey::PVSelection const& sel)
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
	    view.get_parent<Squey::PVSource>().get_format().get_axes().at(c).get_name();

	return new PVGuiQt::PVStatsModel("Count", col1_name, QString(), std::move(col1_out),
	                                 std::move(col2_out), std::move(abs_max), std::move(minmax));
}

}
