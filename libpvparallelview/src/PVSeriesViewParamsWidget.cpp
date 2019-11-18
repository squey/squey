#include <pvparallelview/PVSeriesViewParamsWidget.h>
#include <pvparallelview/PVSeriesViewWidget.h>
#include <pvparallelview/PVSeriesView.h>
#include <pvparallelview/PVSeriesViewZoomer.h>
#include <pvparallelview/PVDisplayViewTimeseries.h>

#include <pvparallelview/common.h>

#include <inendi/PVRangeSubSampler.h>
#include <inendi/widgets/PVAxisComboBox.h>

#include <QSignalMapper>
#include <QShortcut>
#include <QToolButton>
#include <QAction>

PVParallelView::PVSeriesViewParamsWidget::PVSeriesViewParamsWidget(PVCol abscissa,
                                                                   PVSeriesViewWidget* parent)
    : /*QToolBar(parent),*/ _series_view_widget(parent)
{
	bool enable = abscissa != PVCol();
	add_abscissa_selector(abscissa);
	addSeparator();
	add_selection_activator(enable);
	add_hunting_activator(enable);
	addSeparator();
	add_rendering_mode_selector();
	add_sampling_mode_selector();
	add_split_selector();

	setStyleSheet("QToolBar {" + frame_qss_bg_color + "}");
	setAutoFillBackground(true);
	adjustSize();
}

void PVParallelView::PVSeriesViewParamsWidget::add_abscissa_selector(PVCol axis)
{
	_abscissa_selector = new PVWidgets::PVAxisComboBox(
	    _series_view_widget->_view->get_axes_combination(),
	    PVWidgets::PVAxisComboBox::AxesShown::OriginalAxes, [this](PVCol axis, PVCombCol) {
		    return PVDisplays::display_view_if<PVDisplays::PVDisplayViewTimeseries>()
		        .abscissa_filter(_series_view_widget->_view, axis);
	    });
	_abscissa_selector->set_current_axis(axis);
	addWidget(_abscissa_selector);
	connect(_abscissa_selector, &PVWidgets::PVAxisComboBox::current_axis_changed,
	        [this](PVCol axis, PVCombCol) {
		        _series_view_widget->set_abscissa(axis);
		        change_abscissa(axis);
	        });
}

void PVParallelView::PVSeriesViewParamsWidget::add_split_selector()
{
	auto split_selector =
	    new PVWidgets::PVAxisComboBox(_series_view_widget->_view->get_axes_combination(),
	                                  PVWidgets::PVAxisComboBox::AxesShown::OriginalAxes,
	                                  [this](PVCol axis, PVCombCol) { return true; });
	split_selector->insertItem(0, "", QVariant::fromValue(PVCol()));
	split_selector->setCurrentIndex(0);
	split_selector->setToolTip("Select which column to split series on");
	addWidget(split_selector);
	connect(split_selector, &PVWidgets::PVAxisComboBox::current_axis_changed,
	        [this, split_selector](PVCol axis, PVCombCol) {
		        if (axis != _abscissa_selector->current_axis()) {
			        _series_view_widget->set_split(axis);
		        } else {
			        split_selector->set_current_axis(PVCol());
		        }
	        });
	connect(_abscissa_selector, &PVWidgets::PVAxisComboBox::current_axis_changed,
	        [split_selector](PVCol axis, PVCombCol) {
		        split_selector->setEnabled(axis != PVCol());
		        if (axis == split_selector->current_axis()) {
			        split_selector->set_current_axis(PVCol());
		        }
	        });
	split_selector->setEnabled(_abscissa_selector->currentData().value<PVCol>() != PVCol());
}

void PVParallelView::PVSeriesViewParamsWidget::add_selection_activator(bool enable)
{
	QAction* sel = new QAction(this);
	sel->setIcon(QIcon(":/zoom-autofit-horizontal"));
	sel->setCheckable(true);
	sel->setChecked(false);
	sel->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	sel->setText("Select interval");
	sel->setToolTip("Activate/deactivate select interval mode");
	addAction(sel);
	connect(sel, &QAction::triggered, [this](bool checked) {
		_series_view_widget->_zoomer->change_selector_mode(
		    checked ? PVSeriesViewZoomer::SelectorMode::Selecting
		            : PVSeriesViewZoomer::SelectorMode::CrossHairs);
	});
	_bind_connections.push_back([sel, this] {
		sel->setEnabled(true);
		connect(_series_view_widget->_zoomer, &PVSeriesViewZoomer::selector_mode_changed,
		        [sel](PVSeriesViewZoomer::SelectorMode, PVSeriesViewZoomer::SelectorMode mode) {
			        sel->setChecked(mode == PVSeriesViewZoomer::SelectorMode::Selecting);
		        });
	});
	if (enable) {
		_bind_connections.back()();
	}
	sel->setEnabled(enable);
}

void PVParallelView::PVSeriesViewParamsWidget::add_hunting_activator(bool enable)
{
	QAction* hunt = new QAction(this);
	hunt->setIcon(QIcon(":/zoom-autofit-both"));
	hunt->setCheckable(true);
	hunt->setChecked(false);
	hunt->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	hunt->setText("Select series");
	hunt->setToolTip("Activate/deactivate select series mode");
	addAction(hunt);
	connect(hunt, &QAction::triggered, [this](bool checked) {
		_series_view_widget->_zoomer->change_selector_mode(
		    checked ? PVSeriesViewZoomer::SelectorMode::Hunting
		            : PVSeriesViewZoomer::SelectorMode::CrossHairs);
	});
	_bind_connections.push_back([hunt, this] {
		hunt->setEnabled(true);
		connect(_series_view_widget->_zoomer, &PVSeriesViewZoomer::selector_mode_changed,
		        [hunt](PVSeriesViewZoomer::SelectorMode, PVSeriesViewZoomer::SelectorMode mode) {
			        hunt->setChecked(mode == PVSeriesViewZoomer::SelectorMode::Hunting);
		        });
	});
	if (enable) {
		_bind_connections.back()();
	}
	hunt->setEnabled(enable);
}

QToolButton* PVParallelView::PVSeriesViewParamsWidget::add_rendering_mode_selector()
{
	setIconSize(QSize(17, 17));

	_rendering_mode_button = new QToolButton(this);
	_rendering_mode_button->setPopupMode(QToolButton::InstantPopup);
	_rendering_mode_button->setIcon(QIcon(":/series-view-lines"));
	_rendering_mode_button->setToolTip(tr("Rendering mode"));

	// Lines always rendering mode
	QAction* forced_lines_mode = new QAction("Lines", this);
	QShortcut* forced_lines_mode_shortcut =
	    new QShortcut(QKeySequence(Qt::Key_L), _series_view_widget);
	connect(forced_lines_mode_shortcut, &QShortcut::activated,
	        [this, forced_lines_mode]() { set_rendering_mode(forced_lines_mode); });
	forced_lines_mode->setIcon(QIcon(":/series-view-linesalways"));
	forced_lines_mode->setToolTip("Points are always connected");
	forced_lines_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	forced_lines_mode->setData((int)PVSeriesView::DrawMode::LinesAlways);
	_rendering_mode_button->addAction(forced_lines_mode);
	connect(forced_lines_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_rendering_mode));

	// Lines rendering mode
	QAction* lines_mode = new QAction("Mixed", this);
	QShortcut* lines_mode_shortcut = new QShortcut(QKeySequence(Qt::Key_M), _series_view_widget);
	connect(lines_mode_shortcut, &QShortcut::activated,
	        [this, lines_mode]() { set_rendering_mode(lines_mode); });
	lines_mode->setShortcut(Qt::Key_M);
	lines_mode->setIcon(QIcon(":/series-view-lines"));
	lines_mode->setToolTip("Points are connected if they are horizontally adjacent");
	lines_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	lines_mode->setData((int)PVSeriesView::DrawMode::Lines);
	_rendering_mode_button->addAction(lines_mode);
	connect(lines_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_rendering_mode));

	// Points rendering mode
	QAction* points_mode = new QAction("Points", this);
	QShortcut* points_mode_shortcut = new QShortcut(QKeySequence(Qt::Key_P), _series_view_widget);
	connect(points_mode_shortcut, &QShortcut::activated,
	        [this, points_mode]() { set_rendering_mode(points_mode); });
	points_mode->setShortcut(Qt::Key_P);
	points_mode->setIcon(QIcon(":/series-view-points"));
	points_mode->setToolTip("Points are never connected");
	points_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	points_mode->setData((int)PVSeriesView::DrawMode::Points);
	_rendering_mode_button->addAction(points_mode);
	connect(points_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_rendering_mode));

	addWidget(_rendering_mode_button);

	return _rendering_mode_button;
}

void PVParallelView::PVSeriesViewParamsWidget::set_rendering_mode(QAction* action)
{
	_rendering_mode = decltype(_rendering_mode)(action->data().toInt());
	int mode_index = _rendering_mode_button->actions().indexOf(action);

	if (_series_view_widget->_plot) {
		set_rendering_mode(_rendering_mode);
		_series_view_widget->_plot->refresh();
	}

	update_mode_selector(_rendering_mode_button, mode_index);
}

void PVParallelView::PVSeriesViewParamsWidget::set_rendering_mode(PVSeriesView::DrawMode mode) const
{
	_series_view_widget->_plot->set_draw_mode(mode);
}

void PVParallelView::PVSeriesViewParamsWidget::set_rendering_mode()
{
	set_rendering_mode((QAction*)sender());
}

QToolButton* PVParallelView::PVSeriesViewParamsWidget::add_sampling_mode_selector()
{
	setIconSize(QSize(17, 17));

	_sampling_mode_button = new QToolButton(this);
	_sampling_mode_button->setPopupMode(QToolButton::InstantPopup);
	_sampling_mode_button->setIcon(QIcon(":/avg_by"));
	_sampling_mode_button->setToolTip(tr("Sampling mode"));

	// Mean sampling mode
	QAction* mean_mode = new QAction("Average", this);
	QShortcut* mean_mode_shortcut = new QShortcut(QKeySequence(Qt::Key_1), _series_view_widget);
	connect(mean_mode_shortcut, &QShortcut::activated,
	        [this, mean_mode]() { set_sampling_mode(mean_mode); });
	mean_mode->setIcon(QIcon(":/avg_by"));
	mean_mode->setToolTip("Each pixel is the average of its horizontal subrange");
	mean_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	mean_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MEAN);
	_sampling_mode_button->addAction(mean_mode);
	connect(mean_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_sampling_mode));

	// Min sampling mode
	QAction* min_mode = new QAction("Min", this);
	QShortcut* min_mode_shortcut = new QShortcut(QKeySequence(Qt::Key_2), _series_view_widget);
	connect(min_mode_shortcut, &QShortcut::activated,
	        [this, min_mode]() { set_sampling_mode(min_mode); });
	min_mode->setIcon(QIcon(":/min_by"));
	min_mode->setToolTip("Each pixel is the minimum of its horizontal subrange");
	min_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	min_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MIN);
	_sampling_mode_button->addAction(min_mode);
	connect(min_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_sampling_mode));

	// Max sampling mode
	QAction* max_mode = new QAction("Max", this);
	QShortcut* max_mode_shortcut = new QShortcut(QKeySequence(Qt::Key_3), _series_view_widget);
	connect(max_mode_shortcut, &QShortcut::activated,
	        [this, max_mode]() { set_sampling_mode(max_mode); });
	max_mode->setIcon(QIcon(":/max_by"));
	max_mode->setToolTip("Each pixel is the maximum of its horizontal subrange");
	max_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	max_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MAX);
	_sampling_mode_button->addAction(max_mode);
	connect(max_mode, &QAction::triggered, this,
	        qOverload<>(&PVSeriesViewParamsWidget::set_sampling_mode));

	addWidget(_sampling_mode_button);

	return _sampling_mode_button;
}

void PVParallelView::PVSeriesViewParamsWidget::set_sampling_mode(QAction* action)
{
	_sampling_mode = decltype(_sampling_mode)(action->data().toInt());
	int mode_index = _sampling_mode_button->actions().indexOf(action);

	if (_series_view_widget->_sampler) {
		set_sampling_mode(_sampling_mode);
		_series_view_widget->_sampler->resubsample();
	}

	update_mode_selector(_sampling_mode_button, mode_index);
}

void PVParallelView::PVSeriesViewParamsWidget::set_sampling_mode(
    Inendi::PVRangeSubSampler::SAMPLING_MODE mode) const
{
	Inendi::PVRangeSubSampler& sampler = *_series_view_widget->_sampler;
	switch (mode) {
	case Inendi::PVRangeSubSampler::SAMPLING_MODE::MEAN:
		sampler.set_sampling_mode<Inendi::PVRangeSubSampler::SAMPLING_MODE::MEAN>();
		break;
	case Inendi::PVRangeSubSampler::SAMPLING_MODE::MIN:
		sampler.set_sampling_mode<Inendi::PVRangeSubSampler::SAMPLING_MODE::MIN>();
		break;
	case Inendi::PVRangeSubSampler::SAMPLING_MODE::MAX:
		sampler.set_sampling_mode<Inendi::PVRangeSubSampler::SAMPLING_MODE::MAX>();
		break;
	}
}

void PVParallelView::PVSeriesViewParamsWidget::set_sampling_mode()
{
	set_sampling_mode((QAction*)sender());
}

void PVParallelView::PVSeriesViewParamsWidget::update_mode_selector(QToolButton* button,
                                                                    int mode_index)
{
	QAction* action = nullptr;

	try {
		// QList::at has assert in DEBUG mode...
		action = button->actions().at(mode_index);
	} catch (...) {
	}

	if (action != nullptr) {
		button->setIcon(action->icon());
	}
}

void PVParallelView::PVSeriesViewParamsWidget::change_abscissa(PVCol)
{
	for (auto& binder : _bind_connections) {
		binder();
	}
	set_sampling_mode(_sampling_mode);
	set_rendering_mode(_rendering_mode);
}
