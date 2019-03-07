#include <pvparallelview/PVSeriesViewParamsWidget.h>
#include <pvparallelview/PVSeriesViewWidget.h>
#include <pvparallelview/PVSeriesView.h>

#include <pvparallelview/common.h>

#include <inendi/PVRangeSubSampler.h>

#include <QSignalMapper>
#include <QToolButton>

PVParallelView::PVSeriesViewParamsWidget::PVSeriesViewParamsWidget(PVSeriesViewWidget* parent)
    : /*QToolBar(parent),*/ _series_view_widget(parent)
{
	add_rendering_mode_selector();
	add_sampling_mode_selector();

	setStyleSheet("QToolBar {" + frame_qss_bg_color + "}");
	setAutoFillBackground(true);
	adjustSize();
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
	forced_lines_mode->setIcon(QIcon(":/series-view-linesalways"));
	forced_lines_mode->setToolTip("Points are always connected");
	forced_lines_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	forced_lines_mode->setData((int)PVSeriesView::DrawMode::LinesAlways);
	_rendering_mode_button->addAction(forced_lines_mode);
	connect(forced_lines_mode, &QAction::triggered, this,
	        &PVSeriesViewParamsWidget::set_rendering_mode);

	// Lines rendering mode
	QAction* lines_mode = new QAction("Mixed", this);
	lines_mode->setIcon(QIcon(":/series-view-lines"));
	lines_mode->setToolTip("Points are connected if they are horizontally adjacent");
	lines_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	lines_mode->setData((int)PVSeriesView::DrawMode::Lines);
	_rendering_mode_button->addAction(lines_mode);
	connect(lines_mode, &QAction::triggered, this, &PVSeriesViewParamsWidget::set_rendering_mode);

	// Points rendering mode
	QAction* points_mode = new QAction("Points", this);
	points_mode->setIcon(QIcon(":/series-view-points"));
	points_mode->setToolTip("Points are never connected");
	points_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	points_mode->setData((int)PVSeriesView::DrawMode::Points);
	_rendering_mode_button->addAction(points_mode);
	connect(points_mode, &QAction::triggered, this, &PVSeriesViewParamsWidget::set_rendering_mode);

	addWidget(_rendering_mode_button);

	return _rendering_mode_button;
}

void PVParallelView::PVSeriesViewParamsWidget::set_rendering_mode()
{
	QAction* action = (QAction*)sender();
	int mode = action->data().toInt();
	int mode_index = _rendering_mode_button->actions().indexOf(action);

	PVSeriesView& plot = *_series_view_widget->_plot;

	plot.setDrawMode((PVSeriesView::DrawMode)mode);
	plot.refresh();

	update_mode_selector(_rendering_mode_button, mode_index);
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
	mean_mode->setIcon(QIcon(":/avg_by"));
	mean_mode->setToolTip("Each pixel is the average of its horizontal subrange");
	mean_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	mean_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MEAN);
	_sampling_mode_button->addAction(mean_mode);
	connect(mean_mode, &QAction::triggered, this, &PVSeriesViewParamsWidget::set_sampling_mode);

	// Min sampling mode
	QAction* min_mode = new QAction("Min", this);
	min_mode->setIcon(QIcon(":/min_by"));
	min_mode->setToolTip("Each pixel is the minimum of its horizontal subrange");
	min_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	min_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MIN);
	_sampling_mode_button->addAction(min_mode);
	connect(min_mode, &QAction::triggered, this, &PVSeriesViewParamsWidget::set_sampling_mode);

	// Max sampling mode
	QAction* max_mode = new QAction("Max", this);
	max_mode->setIcon(QIcon(":/max_by"));
	max_mode->setToolTip("Each pixel is the maximum of its horizontal subrange");
	max_mode->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	max_mode->setData(Inendi::PVRangeSubSampler::SAMPLING_MODE::MAX);
	_sampling_mode_button->addAction(max_mode);
	connect(max_mode, &QAction::triggered, this, &PVSeriesViewParamsWidget::set_sampling_mode);

	addWidget(_sampling_mode_button);

	return _sampling_mode_button;
}

void PVParallelView::PVSeriesViewParamsWidget::set_sampling_mode()
{
	QAction* action = (QAction*)sender();
	int mode = action->data().toInt();
	int mode_index = _sampling_mode_button->actions().indexOf(action);

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

	sampler.resubsample();

	update_mode_selector(_sampling_mode_button, mode_index);
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