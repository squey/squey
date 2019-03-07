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
#include <QKeyEvent>
#include <QStyledItemDelegate>
#include <QPainter>
#include <QScrollBar>

PVParallelView::PVSeriesViewWidget::PVSeriesViewWidget(Inendi::PVView* view,
                                                       PVCombCol axis_comb,
                                                       QWidget* parent /*= nullptr*/)
    : QWidget(parent), _help_widget(this)
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

	struct StyleDelegate : public QStyledItemDelegate {
		StyleDelegate(QWidget* parent = nullptr) : QStyledItemDelegate(parent) {}
		void paint(QPainter* painter,
		           const QStyleOptionViewItem& option,
		           const QModelIndex& index) const override
		{
			auto color = index.model()->data(index, Qt::UserRole).value<SerieListItemData>().color;
			if ((option.state & QStyle::State_Selected)) {
				painter->fillRect(option.rect, color);
				painter->setPen(Qt::black);
				painter->drawText(option.rect,
				                  index.model()->data(index, Qt::DisplayRole).toString());
			} else {
				// painter->fillRect(option.rect, color);
				painter->setPen(color);
				painter->drawText(option.rect,
				                  index.model()->data(index, Qt::DisplayRole).toString());
			}
		}
	};

	QListWidget* timeseries_list_widget = new QListWidget;
	timeseries_list_widget->setFixedWidth(200);
	timeseries_list_widget->setItemDelegate(new StyleDelegate());
	for (PVCol col(0); col < nraw.column_count(); col++) {
		const PVRush::PVAxisFormat& axis = axes_comb.get_axis(col);
		if (axis.get_type().startsWith("number_") or axis.get_type().startsWith("duration")) {
			QListWidgetItem* item = new QListWidgetItem(axis.get_name());
			QColor color(rand() % 156 + 100, rand() % 156 + 100, rand() % 156 + 100);
			item->setData(Qt::UserRole, QVariant::fromValue(SerieListItemData{col, color}));
			item->setBackgroundColor(color);
			timeseries_list_widget->addItem(item);
		}
	}
	timeseries_list_widget->setSelectionMode(QAbstractItemView::MultiSelection);
	// timeseries_list_widget->setAlternatingRowColors(true);

	// for (PVCol i(0); i < timeseries_list_widget->count(); i++) {
	// 	QColor color(rand() % 156 + 100, rand() % 156 + 100, rand() % 156 + 100);
	// 	timeseries_list_widget->item(i)->setForeground(color); // FIXME
	// }

	const std::vector<PVCol>& combination = axes_comb.get_combination();
	for (PVCol i(0); i < timeseries_list_widget->count(); i++) {
		auto item = timeseries_list_widget->item(i);
		PVCol j = item->data(Qt::UserRole).value<SerieListItemData>().col;
		if (std::find(combination.begin(), combination.end(), j) != combination.end()) {
			item->setSelected(true);
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
			auto item_data = item->data(Qt::UserRole).value<SerieListItemData>();
			seriesDrawOrder.push_back({size_t(item_data.col), item_data.color});
			selected_timeseries.emplace(item_data.col);
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

	auto minmax_changed_f = [this, zoomer](const pvcop::db::array& minmax) {
		PVViewZoomer::Zoom zoom = zoomer->currentZoom();
		std::tie(zoom.minX, zoom.maxX) = _sampler->minmax_to_ratio(minmax);
		// Fix negative value generated by QDateTime resolution fixed to the millisecond
		zoom.minX = std::max(zoom.minX, (PVViewZoomer::zoom_f)0.);
		zoomer->resetAndZoomIn(zoom);
	};

	PVWidgets::PVRangeEdit* range_edit =
	    PVWidgets::PVRangeEditFactory::create(_sampler->minmax_time(), minmax_changed_f);

	QObject::connect(zoomer, &PVSeriesViewZoomer::zoomUpdated,
	                 [range_edit, this](PVViewZoomer::Zoom zoom) {
		                 range_edit->set_minmax(_sampler->ratio_to_minmax(zoom.minX, zoom.maxX));
		             });

	QObject::connect(zoomer, &PVSeriesViewZoomer::selectionCommit, [range_edit, &time, &nraw, view,
	                                                                this](PVViewZoomer::Zoom zoom) {
		const pvcop::db::array& minmax = _sampler->ratio_to_minmax(zoom.minX, zoom.maxX);
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

	QListWidget* selected_series_list = new QListWidget;
	selected_series_list->setFixedWidth(timeseries_list_widget->width());
	selected_series_list->adjustSize();

	QObject::connect(
	    zoomer, &PVSeriesViewZoomer::cursorMoved,
	    [selected_series_list, timeseries_list_widget, zoomer, this](QPoint pos) {
		    selected_series_list->clear();
		    for (const QListWidgetItem* item : timeseries_list_widget->selectedItems()) {
			    const PVCol item_col = item->data(Qt::UserRole).value<SerieListItemData>().col;
			    auto& av_ts = _sampler->sampled_timeserie(item_col);
			    constexpr int radius = 10;
			    for (int r = -radius; r < radius; ++r) {
				    auto av_ts_value = av_ts[pos.x() + r] * uint32_t(zoomer->height());
				    auto min_value = Inendi::PVRangeSubSampler::display_type_max_val *
				                     (zoomer->height() - pos.y() - radius - 1);
				    auto max_value = Inendi::PVRangeSubSampler::display_type_max_val *
				                     (zoomer->height() - pos.y() + radius);
				    if (min_value < av_ts_value and av_ts_value < max_value) {
					    QListWidgetItem* selected_item = new QListWidgetItem(item->text());
					    selected_item->setData(Qt::UserRole, item->data(Qt::UserRole));
					    selected_item->setBackground(
					        item->data(Qt::UserRole).value<SerieListItemData>().color);
					    selected_series_list->addItem(selected_item);
					    break;
				    }
			    }
		    }
		    auto count = selected_series_list->count();
		    auto scrollbar = selected_series_list->horizontalScrollBar();
		    selected_series_list->setMaximumHeight(
		        count > 0
		            ? (count + 1) * selected_series_list->sizeHintForRow(0) +
		                  (scrollbar->isVisible() ? scrollbar->height() : 0)
		            : 0);
		});

	QVBoxLayout* layout = new QVBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);

	QHBoxLayout* hlayout = new QHBoxLayout;
	hlayout->setContentsMargins(0, 0, 0, 0);

	hlayout->addWidget(zoomer);
	auto* vlayout = new QVBoxLayout;
	vlayout->addWidget(timeseries_list_widget);
	vlayout->addWidget(selected_series_list);
	hlayout->addLayout(vlayout);

	layout->addLayout(hlayout);
	layout->addWidget(range_edit);
	layout->addWidget(draw_mode_button);

	// Define help
	setFocusPolicy(Qt::StrongFocus);
	_help_widget.hide();

	_help_widget.initTextFromFile("series view's help", ":help-style");
	_help_widget.addTextFromFile(":help-mouse-series-view");
	_help_widget.newColumn();
	_help_widget.addTextFromFile(":help-selection");

	_help_widget.newTable();
	_help_widget.addTextFromFile(":help-application");
	_help_widget.newColumn();
	_help_widget.finalizeText();

	setLayout(layout);
}

void PVParallelView::PVSeriesViewWidget::keyPressEvent(QKeyEvent* event)
{
	if (PVWidgets::PVHelpWidget::is_help_key(event->key())) {
		if (_help_widget.isHidden()) {
			_help_widget.popup(this, PVWidgets::PVTextPopupWidget::AlignTop,
			                   PVWidgets::PVTextPopupWidget::ExpandAll);
		}
		return;
	}

	QWidget::keyPressEvent(event);
}

void PVParallelView::PVSeriesViewWidget::enterEvent(QEvent*)
{
	setFocus(Qt::MouseFocusReason);
}

void PVParallelView::PVSeriesViewWidget::leaveEvent(QEvent*)
{
	clearFocus();
}
