#ifndef PVVIEWSLISTINGWIDGET_H
#define PVVIEWSLISTINGWIDGET_H

#include <QWidget>
#include <QTreeView>

namespace PVInspector {

class PVTabSplitter;
class PVViewsModel;

class PVViewsListingWidget: public QWidget
{
public:
	PVViewsListingWidget(PVTabSplitter* tab);

protected slots:
	void show_ctxt_menu(const QPoint& pt);

protected:
	PVTabSplitter* _tab_parent;
	QTreeView* _tree;
	PVViewsModel* _model;

	Q_OBJECT
};

}

#endif
