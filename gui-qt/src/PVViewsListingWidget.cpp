#include <PVTabSplitter.h>
#include <PVViewsModel.h>
#include <PVViewsListingWidget.h>
#include <PVMappingPlottingEditDialog.h>

#include <QVBoxLayout>

PVInspector::PVViewsListingWidget::PVViewsListingWidget(PVTabSplitter* tab):
	QWidget(tab),
	_tab_parent(tab)
{
	_model = new PVViewsModel(*tab->get_lib_src(), this);
	_tree = new QTreeView();
	_tree->setModel(_model);
	_tree->setHeaderHidden(true);

	connect(_tree, SIGNAL(customContextMenuRequested(const QPoint&)), this, SLOT(show_ctxt_menu(const QPoint&)));
	_tree->setContextMenuPolicy(Qt::CustomContextMenu);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addWidget(_tree);
	setLayout(layout);

	setFocusPolicy(Qt::StrongFocus);
}

void PVInspector::PVViewsListingWidget::show_ctxt_menu(const QPoint& pt)
{
	QModelIndex idx_click = _tree->indexAt(pt);
	PVViewsModel::PVIndexNode node_obj = _model->get_object(idx_click);

	// Create menu
	QMenu* ctxt_menu = new QMenu();
	QAction* edit_action = new QAction(tr("Edit..."), NULL);
	ctxt_menu->addAction(edit_action);

	// Show it
	QAction* sel_action = ctxt_menu->exec(QCursor::pos());
	if (sel_action == edit_action) {
		PVMappingPlottingEditDialog* dlg;
		if (node_obj.is_mapped()) {
			dlg = new PVMappingPlottingEditDialog(&node_obj.as_mapped()->get_mapping(), NULL, this);
		}
		else {
			dlg = new PVMappingPlottingEditDialog(NULL, &node_obj.as_plotted()->get_plotting(), this);
		}
		dlg->exec();
	}
}
