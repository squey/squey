#ifndef PVVIEWSLISTINGWIDGET_H
#define PVVIEWSLISTINGWIDGET_H

#include <QWidget>
#include <QTreeView>

namespace PVInspector {

class PVTabSplitter;
class PVViewsModel;
class PVViewsListingView;

class PVViewsListingWidget: public QWidget
{
public:
	PVViewsListingWidget(PVTabSplitter* tab);

public:
	void force_refresh();
	PVViewsListingView* get_view() { return _tree; }
	PVViewsModel* get_model() { return _model; }

protected slots:
	void show_ctxt_menu(const QPoint& pt);

protected:
	PVTabSplitter* _tab_parent;
	PVViewsListingView* _tree;
	PVViewsModel* _model;

	Q_OBJECT
};

}

#endif
