
#include <picviz/PVView.h>

#include <pvparallelview/PVAxisLabel.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVLibView.h>

#include <QDialog>
#include <QLayout>
#include <QMenu>

#include <iostream>

/*****************************************************************************
 * PVParallelView::PVAxisLabel::contextMenuEvent
 *****************************************************************************/

void PVParallelView::PVAxisLabel::contextMenuEvent(QGraphicsSceneContextMenuEvent *event)
{
	QMenu menu;

	QAction *azv = menu.addAction("New zoomed view");
	connect(azv, SIGNAL(triggered()), this, SLOT(new_zoomed_parallel_view()));

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
	QDialog *view_dlg = new QDialog();

	view_dlg->setMaximumWidth(1024);
	view_dlg->setMaximumHeight(1024);
	view_dlg->setAttribute(Qt::WA_DeleteOnClose, true);

	QLayout *view_layout = new QVBoxLayout(view_dlg);
	view_layout->setContentsMargins(0, 0, 0, 0);
	view_dlg->setLayout(view_layout);

	QWidget *view = common::get_lib_view(const_cast<Picviz::PVView&>(_lib_view))->create_zoomed_view(_axis_index);

	view_layout->addWidget(view);
	view_dlg->show();
}

/*****************************************************************************
 * PVParallelView::PVAxisLabel::new_selection_sliders
 *****************************************************************************/

void PVParallelView::PVAxisLabel::new_selection_sliders()
{
	_sliders_group->add_selection_sliders(0, 1024);
}
