#include <pvkernel/core/qobject_helpers.h>
#include <picviz/PVView.h>

#include <pvparallelview/PVAxisLabel.h>
#include <pvparallelview/PVSlidersGroup.h>

#include <QDialog>
#include <QLayout>
#include <QMenu>
#include <QGraphicsScene>
#include <QGraphicsView>

#include <iostream>

#include <pvdisplays/PVDisplaysImpl.h>
#include <pvdisplays/PVDisplaysContainer.h>

/*****************************************************************************
 * PVParallelView::PVAxisLabel::PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::PVAxisLabel(const Picviz::PVView &view,
                                         PVSlidersGroup *sg,
                                         QGraphicsItem *parent) :
	QGraphicsSimpleTextItem(parent), _lib_view(view), _sliders_group(sg)
{
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::~PVAxisLabel
 *****************************************************************************/

PVParallelView::PVAxisLabel::~PVAxisLabel()
{
	if (scene()) {
		scene()->removeItem(this);
	}
	if (group()) {
		group()->removeFromGroup(this);
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::contextMenuEvent
 *****************************************************************************/

void PVParallelView::PVAxisLabel::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
	// Get parent PVDisplaysContainer if available
	QList<QGraphicsView*> parent_views = scene()->views();
	assert(parent_views.size() == 1);

	QWidget* parent_view = parent_views.at(0);
	PVDisplays::PVDisplaysContainer* container = PVCore::get_qobject_parent_of_type<PVDisplays::PVDisplaysContainer*>(parent_view);

	QMenu menu;

	/*
	QAction *azv = menu.addAction("New zoomed view");
	connect(azv, SIGNAL(triggered()), this, SLOT(new_zoomed_parallel_view()));*/

	if (container) {
		PVDisplays::get().add_displays_view_axis_menu(menu, container, SLOT(create_view_axis_widget()), (Picviz::PVView*) &_lib_view, get_axis_index());
		menu.addSeparator();
	}
	QAction *ars = menu.addAction("New selection cursors");
	connect(ars, SIGNAL(triggered()), this, SLOT(new_selection_sliders()));

	if (menu.exec(event->screenPos()) != nullptr) {
		event->accept();
	}
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::new_zoomed_parallel_view
 *****************************************************************************/

void PVParallelView::PVAxisLabel::new_zoomed_parallel_view()
{
	/*QDialog *view_dlg = new QDialog();

	view_dlg->setMaximumWidth(1024);
	view_dlg->setMaximumHeight(1024);
	view_dlg->setAttribute(Qt::WA_DeleteOnClose, true);

	QLayout *view_layout = new QVBoxLayout(view_dlg);
	view_layout->setContentsMargins(0, 0, 0, 0);
	view_dlg->setLayout(view_layout);

	QWidget *view = common::get_lib_view(const_cast<Picviz::PVView&>(_lib_view))->create_zoomed_view(_axis_index);

	view_layout->addWidget(view);
	view_dlg->show();*/

	emit new_zoomed_parallel_view(_axis_index);
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVAxisLabel::new_selection_sliders()
{
	_sliders_group->add_selection_sliders(0, 1024);
}

PVCol PVParallelView::PVAxisLabel::get_axis_index() const
{
	return _lib_view.get_axes_combination().get_index_by_id(_sliders_group->get_axis_id()); 
}
