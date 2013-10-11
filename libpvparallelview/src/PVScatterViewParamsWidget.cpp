
#include <pvparallelview/PVScatterViewParamsWidget.h>
#include <pvparallelview/PVScatterViewSelectionRectangle.h>
#include <pvparallelview/PVScatterView.h>

#include <QSignalMapper>
#include <QToolBar>
#include <QToolButton>

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget
 *****************************************************************************/

PVParallelView::PVScatterViewParamsWidget::PVScatterViewParamsWidget(PVScatterView* parent):
	QToolBar(parent)
{

	_sel_mode_signal_mapper = new QSignalMapper(this);
	QObject::connect(_sel_mode_signal_mapper, SIGNAL(mapped(int)),
	                 this, SLOT(set_selection_mode(int)));

	_sel_mode_button = PVSelectionRectangle::add_selection_mode_selector(parent,
	                                                                     this,
	                                                                     _sel_mode_signal_mapper);
}

/*****************************************************************************
 * PVParallelView::PVScatterViewParamsWidget::set_selection_mode
 *****************************************************************************/

void PVParallelView::PVScatterViewParamsWidget::set_selection_mode(int mode)
{
	PVSelectionRectangle::update_selection_mode_selector(_sel_mode_button,
	                                                     mode);

	PVScatterView *sv = parent_sv();
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
