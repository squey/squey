/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#include <pvparallelview/PVSeriesViewWidget.h>

#include <pvparallelview/PVSeriesView.h>
#include <pvparallelview/PVSeriesViewZoomer.h>
#include <pvkernel/widgets/PVRangeEdit.h>
#include <pvkernel/rush/PVNraw.h>
#include <inendi/PVSource.h>
#include <inendi/PVRangeSubSampler.h>

#include <QListWidget>

PVParallelView::PVSeriesViewWidget::PVSeriesViewWidget(Inendi::PVView* view,
                                                       PVCombCol axis_comb,
                                                       QWidget* parent /*= nullptr*/)
    : QWidget(parent)
{
	auto plotteds = view->get_parent<Inendi::PVSource>().get_children<Inendi::PVPlotted>();
	const Inendi::PVAxesCombination& axes_comb = view->get_axes_combination();
	PVCol col = axes_comb.get_nraw_axis(axis_comb);
	PVRush::PVNraw const& nraw = view->get_rushnraw_parent();
	const auto& plotteds_vector = plotteds.front()->get_plotteds();

	const pvcop::db::array& time = nraw.column(col);

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (PVCol i : axes_comb.get_combination()) {
		timeseries.emplace_back(plotteds_vector[i].to_core_array<uint32_t>());
	}

	_sampler.reset(new Inendi::PVRangeSubSampler(time, timeseries));
	_sampler->subsample();

	PVSeriesView* plot = new PVSeriesView(*_sampler);
	plot->setBackgroundColor(QColor(10, 10, 10, 255));

	QListWidget* timeseries_list_widget = new QListWidget;
	timeseries_list_widget->setFixedWidth(200);
	timeseries_list_widget->addItems(axes_comb.get_combined_names());
	timeseries_list_widget->setSelectionMode(QAbstractItemView::MultiSelection);
	timeseries_list_widget->setAlternatingRowColors(true);

	{
		std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder;
		for (PVCombCol i(0); i < timeseries.size(); i++) {
			QColor color(rand() % 256, rand() % 256, rand() % 256);
			if (axes_comb.get_axis(i).get_type().left(6) == "number") {
				timeseries_list_widget->item(i)->setSelected(true);
				seriesDrawOrder.push_back({i, color});
			}
			timeseries_list_widget->item(i)->setForeground(color); // FIXME
		}
		plot->showSeries(std::move(seriesDrawOrder));
	}

	QObject::connect(timeseries_list_widget, &QListWidget::itemSelectionChanged,
	                 [ plot, timeseries_list_widget, timeseries_size = timeseries.size() ]() {
		                 // FIXME : should put newly selected timeserie on top
		                 std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder;
		                 for (PVCombCol i(0); i < timeseries_size; ++i) {
			                 if (timeseries_list_widget->item(i)->isSelected()) {
				                 seriesDrawOrder.push_back(
				                     {i, timeseries_list_widget->item(i)->foreground().color()});
			                 }
		                 }
		                 plot->showSeries(std::move(seriesDrawOrder));
		                 plot->update();
		             });

	PVSeriesViewZoomer* zoomer = new PVSeriesViewZoomer(plot, *_sampler);
	zoomer->setZoomRectColor(Qt::red);

	auto minmax_changed_f = [this, plot](const pvcop::db::array& minmax) {
		_sampler->subsample(minmax);
		plot->update();
	};

	PVWidgets::PVRangeEdit* range_edit = nullptr;
	if (_sampler->minmax_time().formatter()->name().find("datetime") == 0) {
		range_edit = new PVWidgets::PVDateTimeRangeEdit(_sampler->minmax_time(), minmax_changed_f);
	} else if (_sampler->minmax_time().formatter()->name().find("number_uint") == 0) {
		range_edit = new PVWidgets::PVIntegerRangeEdit(_sampler->minmax_time(), minmax_changed_f);
	}

	view->get_parent<Inendi::PVPlotted>()._plotted_updated.connect([this, plot]() {
		_sampler->resubsample();
		plot->update();
	});

	QVBoxLayout* layout = new QVBoxLayout;

	QHBoxLayout* hlayout = new QHBoxLayout;

	hlayout->addWidget(zoomer);
	hlayout->addWidget(timeseries_list_widget);

	layout->addLayout(hlayout);
	layout->addWidget(range_edit);

	setLayout(layout);
}
