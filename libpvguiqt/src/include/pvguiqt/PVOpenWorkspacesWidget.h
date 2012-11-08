#ifndef PVGUIQT_PVOPENWORKSPACES_WIDGET_H
#define PVGUIQT_PVOPENWORKSPACES_WIDGET_H

#include <picviz/PVRoot_types.h>
#include <QWidget>

namespace PVGuiQt {

class PVOpenWorkspacesTabWidget;
class PVRootTreeView;

class PVOpenWorkspacesWidget: public QWidget
{
	Q_OBJECT

public:
	PVOpenWorkspacesWidget(Picviz::PVRoot* root, QWidget* parent = NULL);

public:
	inline PVOpenWorkspacesTabWidget* workspace_tab_widget() const { return _tab_widget; }

private slots:
	void create_views_widget();

private:
	PVOpenWorkspacesTabWidget* _tab_widget;
	PVRootTreeView* _root_view;
};

}

#endif
