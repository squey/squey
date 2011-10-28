#include <PVViewsListingView.h>
#include <PVViewsModel.h>
#include <PVTabSplitter.h>

PVInspector::PVViewsListingView::PVViewsListingView(PVViewsModel* model, PVTabSplitter* tab, QWidget* parent):
	QTreeView(parent),
	_tab(tab),
	_model(model)
{
	setModel(model);
	setMinimumSize(0,0);
	setHeaderHidden(true);
	setAllColumnsShowFocus(true);
}

void PVInspector::PVViewsListingView::mouseDoubleClickEvent(QMouseEvent* event)
{
	QTreeView::mouseDoubleClickEvent(event);

	QModelIndex idx_click = indexAt(event->pos());
	if (!idx_click.isValid()) {
		PVLOG_INFO("index not valid\n");
		return;
	}

	PVViewsModel::PVIndexNode node_obj(_model->get_object(idx_click));
	if (node_obj.is_plotted()) {
		_tab->select_plotted(node_obj.as_plotted());
		_model->emitDataChanged(idx_click);
	}
}
