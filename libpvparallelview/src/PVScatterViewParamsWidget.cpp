/**
 * @file
 *
 * @copyright (C) Picviz Labs 2013-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvparallelview/PVScatterViewParamsWidget.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterView.h>

#include <QSignalMapper>
#include <QToolBar>
#include <QToolButton>
#include <QAction>

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget
 *****************************************************************************/

PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget(PVScatterView* parent)
    : QToolBar(parent)
{
	_sel_mode_signal_mapper = new QSignalMapper(this);
	QObject::connect(_sel_mode_signal_mapper,
	                 &QSignalMapper::mappedInt, this,
	                 &PVScatterViewParamsWidget::set_selection_mode);

	_sel_mode_button =
	    PVSelectionRectangle::add_selection_mode_selector(parent, this, _sel_mode_signal_mapper);

	_show_labels = new QAction(this);
	_show_labels->setIcon(QIcon(":/labeled-axis"));
	_show_labels->setCheckable(true);
	_show_labels->setChecked(false);
	_show_labels->setShortcut(Qt::Key_T);
	_show_labels->setText("Toggle labels visibility");
	_show_labels->setShortcutContext(Qt::WidgetWithChildrenShortcut);
	_show_labels->setToolTip("Activate/deactivate labels display on axes (" +
	                         _show_labels->shortcut().toString() + ")");
	addAction(_show_labels);
	parent->addAction(_show_labels);
	connect(_show_labels, &QAction::toggled, parent_sv(), &PVScatterView::toggle_show_labels);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::update_widgets
 *****************************************************************************/

void PVParallelView::PVScatterViewParamsWidget::update_widgets()
{
	_show_labels->blockSignals(true);

	_show_labels->setChecked(parent_sv()->show_labels());

	_show_labels->blockSignals(false);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVScatterViewParamsWidget::set_selection_mode(int mode)
{
	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button, mode);

	PVScatterView* sv = parent_sv();
	sv->get_selection_rect()->set_selection_mode(mode);
	sv->fake_mouse_move();
	sv->get_viewport()->update();
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::parent_sv
 *****************************************************************************/

PVParallelView::PVScatterView* PVParallelView::PVScatterViewParamsWidget::parent_sv()
{
	assert(qobject_cast<PVScatterView*>(parent()));
	return static_cast<PVScatterView*>(parent());
}
