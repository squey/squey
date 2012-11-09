#include <pvdisplays/PVDisplaysImpl.h>
#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVOpenWorkspacesWidget.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVRootTreeModel.h>
#include <pvguiqt/PVRootTreeView.h>

#include <picviz/PVRoot.h>
#include <picviz/PVView.h>

#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QToolBar>
#include <QToolButton>
#include <QSplitter>

namespace PVGuiQt { namespace __impl {

class RootTreeModelViewsSelectable: public PVGuiQt::PVRootTreeModel
{
public:
	RootTreeModelViewsSelectable(PVCore::PVDataTreeObjectBase& root, QObject* parent = 0):
		PVGuiQt::PVRootTreeModel(root, parent)
	{ }

public:
	Qt::ItemFlags flags(const QModelIndex & index) const
	{
		PVCore::PVDataTreeObjectBase const* obj = (PVCore::PVDataTreeObjectBase const*) index.internalPointer();
		Picviz::PVView const* view = dynamic_cast<Picviz::PVView const*>(obj);
		Qt::ItemFlags flags = PVGuiQt::PVRootTreeModel::flags(index);
		if (view) {
			flags |= Qt::ItemIsSelectable;
		}
		else {
			flags &= ~Qt::ItemIsSelectable;
		}
		return flags;
	}
};

#if 0
class RootTreeProxyModel: public QAbstractProxyModel
{
protected:
	QModelIndex mapFromSource(QModelIndex const& src_idx) const override
	{
		PVCore::PVDataTreeObjectBase const* obj = (PVCore::PVDataTreeObjectBase const*) index.internalPointer();
		if (dynamic_cast<Picviz::PVMapped const*>(obj) || dynamic_cast<Picviz::PVPlotted const*>(obj)) {
			return QModelIndex();
		}

		QModelIndex parent = src_idx.parent();
		Picviz::PVView const* view = dynamic_cast<Picviz::PVView const*>(obj);
		if (view) {
			parent = qobject_cast<PVWidgets::PVDataTreeModel*>(sourceModel())->index_from_obj(view->get_parent<Picviz::PVScene*>(view));
		}

		return index(
	}

	QModelIndex mapToSource(QModelIndex const& idx) const override
	{
		PVCore::PVDataTreeObjectBase const* obj = (PVCore::PVDataTreeObjectBase const*) index.internalPointer();
		return qobject_cast<PVWidgets::PVDataTreeModel*>(sourceModel())->index_from_obj(obj);
	}
};
#endif

} }

PVGuiQt::PVOpenWorkspacesWidget::PVOpenWorkspacesWidget(Picviz::PVRoot* root, QWidget* parent):
	QWidget(parent)
{
	// Layouts
	//
	QHBoxLayout* main_layout = new QHBoxLayout();

	// to make projects and workspaces tab aligned
	main_layout->setContentsMargins(0, 0, 0, 0);

	QVBoxLayout* left_layout = new QVBoxLayout();

	// to make left widgets be aligned with the right widgets
	left_layout->setContentsMargins(0, 0, 0, 0);
	left_layout->setSpacing(0);

	// Widgets
	//
	
	QSplitter* main_splitter = new QSplitter(Qt::Horizontal, this);

	// Open workspaces
	_tab_widget = new PVOpenWorkspacesTabWidget(*root);

	// Data tree from PVRoot
	__impl::RootTreeModelViewsSelectable* tree_model = new __impl::RootTreeModelViewsSelectable(*root);
	_root_view = new PVRootTreeView(tree_model);
	_root_view->setContextMenuPolicy(Qt::NoContextMenu); // Disable data-tree creation context menu
	_root_view->setSelectionMode(QAbstractItemView::ExtendedSelection);
	connect(tree_model, SIGNAL(modelReset()), _root_view, SLOT(expandAll()));

	// View creation tab bar
	QToolBar* toolbar = new QToolBar();
	PVDisplays::get().visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
		[&](PVDisplays::PVDisplayViewIf& obj)
		{
			if (!obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters)) {
				QToolButton* btn = new QToolButton(toolbar);

				QAction* act = new QAction(btn);
				act->setIcon(obj.toolbar_icon());
				act->setToolTip(obj.tooltip_str());

				QVariant var;
				var.setValue<void*>(&obj);
				act->setData(var);

				connect(act, SIGNAL(triggered()), this, SLOT(create_views_widget()));

				btn->setDefaultAction(act);
				toolbar->addWidget(btn);

			}
		}, PVDisplays::PVDisplayIf::ShowInToolbar);


	// Composition of everyone
	//
	
	left_layout->addWidget(toolbar);
	left_layout->addWidget(_root_view);
	QWidget* left_widget = new QWidget(this);
	left_widget->setAutoFillBackground(true);
	left_widget->setLayout(left_layout);

	main_splitter->addWidget(left_widget);
	main_splitter->addWidget(_tab_widget);

	// Workspaces tab widget isn't collapsible
	main_splitter->setCollapsible(1, false);
	QList<int> sizes;
	sizes << 1 << 1;
	main_splitter->setSizes(sizes);

	main_layout->addWidget(main_splitter);

	setLayout(main_layout);
}

void PVGuiQt::PVOpenWorkspacesWidget::create_views_widget()
{
	QAction* act = qobject_cast<QAction*>(sender());
	if (!act) {
		return;
	}

	PVOpenWorkspace* cur_workspace = _tab_widget->current_workspace_or_create();

	PVDisplays::PVDisplayViewIf& interface = *(reinterpret_cast<PVDisplays::PVDisplayViewIf*>(act->data().value<void*>()));
	_root_view->visit_selected_objs_as<Picviz::PVView>(
		[&](Picviz::PVView* view)
		{
			QAction* creation_act = PVDisplays::get().action_bound_to_params(interface, view);
			cur_workspace->create_view_widget(creation_act);
		});
}
