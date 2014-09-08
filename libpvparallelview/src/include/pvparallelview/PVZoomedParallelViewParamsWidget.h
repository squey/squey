
#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

#include <pvbase/types.h>

class QStringList;
class QMenu;
class QToolButton;

namespace PVParallelView
{
class PVZoomedParallelView;

class PVZoomedParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

public:
	PVZoomedParallelViewParamsWidget(PVZoomedParallelView* parent);

public:
	void build_axis_menu(int active_axis, const QStringList& sl);

signals:
	void change_to_col(int new_axis);

private slots:
	void set_active_axis_action(QAction *act);

private:
	QToolButton *_menu_toolbutton;
	QMenu       *_axes;
	QAction     *_active_axis_action;
	PVCol        _active_axis;
};

}

#endif // PVPARALLELVIEW_ZOOMEDPARALLELVIEWPARAMSWIDGET_H

