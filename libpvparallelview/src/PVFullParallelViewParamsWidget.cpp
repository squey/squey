/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2019
 */

#include <pvparallelview/PVFullParallelViewParamsWidget.h>
#include <pvparallelview/PVFullParallelViewSelectionRectangle.h>
#include <pvparallelview/PVFullParallelView.h>

#include <QVBoxLayout>
#include <QToolBar>
#include <QCheckBox>
#include <QSignalMapper>
#include <QMenu>

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget
 *****************************************************************************/

PVParallelView::PVFullParallelViewParamsWidget::PVFullParallelViewParamsWidget(
    PVFullParallelView* parent)
    : QToolBar(parent)
{
	_signal_mapper = new QSignalMapper(this);
	QObject::connect(_signal_mapper,
	                 static_cast<void (QSignalMapper::*)(int)>(&QSignalMapper::mapped), this,
	                 &PVFullParallelViewParamsWidget::set_selection_mode);

	_sel_mode_button =
	    PVSelectionRectangle::add_selection_mode_selector(parent, this, _signal_mapper);

	addSeparator();
	setVisible(false);

	// _autofit = new QAction(this);
	// _autofit->setIcon(QIcon(":/zoom-autofit-horizontal"));
	// _autofit->setCheckable(true);
	// _autofit->setChecked(false);
	// _autofit->setShortcut(Qt::Key_F);
	// _autofit->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	// _autofit->setText("View auto-fit on selected events");
	// _autofit->setToolTip("Activate/deactivate horizontal auto-fit on selected events (" +
	//                      _autofit->shortcut().toString() + ")");
	// addAction(_autofit);
	// parent->addAction(_autofit);
	// connect(_autofit, &QAction::toggled, parent_hcv(),
	// &PVFullParallelView::toggle_auto_x_zoom_sel);

	// _use_log_color = new QAction("Logarithmic colormap", this);
	// _use_log_color->setIcon(QIcon(":/colormap-log"));
	// _use_log_color->setCheckable(true);
	// _use_log_color->setChecked(false);
	// _use_log_color->setShortcut(Qt::Key_L);
	// _use_log_color->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	// _use_log_color->setText("Logarithmic colormap");
	// _use_log_color->setToolTip(
	//     "Activate/deactivate use of a logarithmic colormap for visible events (" +
	//     _use_log_color->shortcut().toString() + ")");
	// addAction(_use_log_color);
	// parent->addAction(_use_log_color);
	// connect(_use_log_color, &QAction::toggled, parent_hcv(),
	// &PVFullParallelView::toggle_log_color);

	// _show_labels = new QAction(this);
	// _show_labels->setIcon(QIcon(":/labeled-axis"));
	// _show_labels->setCheckable(true);
	// _show_labels->setChecked(false);
	// _show_labels->setShortcut(Qt::Key_T);
	// _show_labels->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	// _show_labels->setText("Toggle labels visibility");
	// _show_labels->setToolTip("Activate/deactivate labels display on axes (" +
	//                          _show_labels->shortcut().toString() + ")");
	// addAction(_show_labels);
	// parent->addAction(_show_labels);
	// connect(_show_labels, &QAction::toggled, parent_hcv(),
	// &PVFullParallelView::toggle_show_labels);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::update_widgets()
{
	// _autofit->blockSignals(true);
	// _use_log_color->blockSignals(true);
	// _show_labels->blockSignals(true);

	// _autofit->setChecked(parent_hcv()->auto_x_zoom_sel());
	// _use_log_color->setChecked(parent_hcv()->use_log_color());
	// _show_labels->setChecked(parent_hcv()->show_labels());

	// _autofit->blockSignals(false);
	// _use_log_color->blockSignals(false);
	// _show_labels->blockSignals(false);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVFullParallelViewParamsWidget::set_selection_mode(int mode)
{
	// PVFullParallelView* hcv = parent_hcv();
	// hcv->get_selection_rect()->set_selection_mode(mode);
	// hcv->fake_mouse_move();
	// hcv->get_viewport()->update();

	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);
}

/*****************************************************************************
 * PVParallelView::PVFullParallelViewParamsWidget::parent_hcv
 *****************************************************************************/

PVParallelView::PVFullParallelView* PVParallelView::PVFullParallelViewParamsWidget::parent_hcv()
{
	assert(qobject_cast<PVFullParallelView*>(parentWidget()));
	return static_cast<PVFullParallelView*>(parentWidget());
}
