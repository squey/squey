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
#include <QStateMachine>

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
	for (PVCol col(0); col < nraw.column_count(); col++) {
		timeseries.emplace_back(plotteds_vector[col].to_core_array<uint32_t>());
	}

	_sampler.reset(
	    new Inendi::PVRangeSubSampler(time, timeseries, nraw, view->get_real_output_selection()));

	PVSeriesView* plot = new PVSeriesView(*_sampler, PVSeriesView::Backend::Default);
	plot->setBackgroundColor(QColor(10, 10, 10, 255));

	QListWidget* timeseries_list_widget = new QListWidget;
	timeseries_list_widget->setFixedWidth(200);
	for (PVCol col(0); col < nraw.column_count(); col++) {
		const PVRush::PVAxisFormat& axis = axes_comb.get_axis(col);
		if (axis.get_type().startsWith("number_") or axis.get_type().startsWith("duration")) {
			QListWidgetItem* item = new QListWidgetItem(axis.get_name());
			item->setData(Qt::UserRole, QVariant(col.value()));
			timeseries_list_widget->addItem(item);
		}
	}
	timeseries_list_widget->setSelectionMode(QAbstractItemView::MultiSelection);
	timeseries_list_widget->setAlternatingRowColors(true);

	for (PVCol i(0); i < timeseries_list_widget->count(); i++) {
		QColor color(rand() % 156 + 100, rand() % 156 + 100, rand() % 156 + 100);
		timeseries_list_widget->item(i)->setForeground(color); // FIXME
	}

	const std::vector<PVCol>& combination = axes_comb.get_combination();
	for (PVCol i(0); i < timeseries_list_widget->count(); i++) {
		PVCol j(timeseries_list_widget->item(i)->data(Qt::UserRole).toUInt());
		if (std::find(combination.begin(), combination.end(), j) != combination.end()) {
			timeseries_list_widget->item(i)->setSelected(true);
		}
	}

	auto update_selected_timeseries =
	    [ plot, timeseries_list_widget, timeseries_size = timeseries_list_widget->count(),
		  this ](bool resample = true)
	{
		// FIXME : should put newly selected timeserie on top
		std::vector<PVSeriesView::SerieDrawInfo> seriesDrawOrder;
		std::unordered_set<size_t> selected_timeseries;
		selected_timeseries.reserve(timeseries_size);
		for (const QListWidgetItem* item : timeseries_list_widget->selectedItems()) {
			const int item_col = item->data(Qt::UserRole).toInt();
			seriesDrawOrder.push_back({size_t(item_col), item->foreground().color()});
			selected_timeseries.emplace(item_col);
		}
		_sampler->set_selected_timeseries(selected_timeseries);
		if (resample) {
			_sampler->resubsample();
		}
		plot->showSeries(std::move(seriesDrawOrder));
		plot->update();
	};

	QObject::connect(timeseries_list_widget, &QListWidget::itemSelectionChanged,
	                 update_selected_timeseries);
	update_selected_timeseries(false);

	PVSeriesViewZoomer* zoomer = new PVSeriesViewZoomer(plot, *_sampler);
	zoomer->setZoomRectColor(Qt::red);

	QPushButton* draw_mode_button = new QPushButton();
	{
		QStateMachine* qsm = new QStateMachine(draw_mode_button);
		std::vector<QState*> states;
		auto add_state = [plot, qsm, draw_mode_button, &states](PVSeriesView::DrawMode mode,
		                                                        QString text) -> QState* {
			if (plot->capability(mode) == mode) {
				QState* state = new QState(qsm);
				state->assignProperty(draw_mode_button, "text", std::move(text));
				connect(state, &QState::entered, [plot, mode] {
					plot->setDrawMode(mode);
					plot->refresh();
				});
				states.push_back(state);
				return state;
			}
			return nullptr;
		};
		add_state(PVSeriesView::DrawMode::Lines, "Lines");
		add_state(PVSeriesView::DrawMode::Points, "Points");
		add_state(PVSeriesView::DrawMode::LinesAlways, "Lines Always");
		for (size_t i = 0; i < states.size(); ++i) {
			states[i]->addTransition(draw_mode_button, &QPushButton::clicked,
			                         states[(i + 1) % states.size()]);
		}
		qsm->setInitialState(states.front());
		qsm->start();
	}

	auto minmax_changed_f = [this](const pvcop::db::array& minmax) { _sampler->subsample(minmax); };

	PVWidgets::PVRangeEdit* range_edit = nullptr; // TODO : use a factory
	if (_sampler->minmax_time().formatter()->name().find("datetime") == 0) {
		range_edit = new PVWidgets::PVDateTimeRangeEdit(_sampler->minmax_time(), minmax_changed_f);
	} else if (_sampler->minmax_time().formatter()->name().find("number_float") == 0 or
	           _sampler->minmax_time().formatter()->name().find("number_double") == 0) {
		range_edit = new PVWidgets::PVDoubleRangeEdit(_sampler->minmax_time(), minmax_changed_f);
	} else if (_sampler->minmax_time().formatter()->name().find("number_uint") == 0) {
		range_edit = new PVWidgets::PVIntegerRangeEdit(_sampler->minmax_time(), minmax_changed_f);
	}

	QObject::connect(zoomer, &PVSeriesViewZoomer::zoomUpdated,
	                 [range_edit, this](PVViewZoomer::Zoom zoom) {
		                 range_edit->set_minmax(_sampler->minmax_subrange(zoom.minX, zoom.maxX));
		             });

	QObject::connect(zoomer, &PVSeriesViewZoomer::selectionCommit, [range_edit, &time, &nraw, view,
	                                                                this](PVViewZoomer::Zoom zoom) {
		const pvcop::db::array& minmax = _sampler->minmax_subrange(zoom.minX, zoom.maxX);
		range_edit->set_minmax(minmax);
		const auto& sorted_indexes = _sampler->sorted_indexes();
		pvcop::db::range_t selected_range = time.equal_range(minmax, sorted_indexes);
		const auto& sort = sorted_indexes ? sorted_indexes.to_core_array()
		                                  : pvcop::core::array<pvcop::db::index_t>();
		Inendi::PVSelection sel(nraw.row_count());
		sel.select_none(); // Not sur if needed
		for (size_t i = selected_range.begin; i < selected_range.end; i++) {
			sel.set_bit_fast(sort ? sort[i] : i);
		}
		view->set_selection_view(sel);
	});

	// Subscribe to plotting changes
	_plotting_change_connection = view->get_parent<Inendi::PVPlotted>()._plotted_updated.connect(
	    [this](const QList<PVCol>& plotteds_updated) {
		    std::unordered_set<size_t> updated_timeseries(plotteds_updated.begin(),
		                                                  plotteds_updated.end());
		    _sampler->resubsample(updated_timeseries);
		});

	// Subscribe to selection changes
	_selection_change_connection =
	    view->_update_output_selection.connect([this]() { _sampler->resubsample(); });

	QVBoxLayout* layout = new QVBoxLayout;

	QHBoxLayout* hlayout = new QHBoxLayout;

	hlayout->addWidget(zoomer);
	hlayout->addWidget(timeseries_list_widget);

	layout->addLayout(hlayout);
	layout->addWidget(range_edit);
	layout->addWidget(draw_mode_button);

	setLayout(layout);
}
