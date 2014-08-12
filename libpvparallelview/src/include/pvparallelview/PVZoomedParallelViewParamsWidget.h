
#ifndef PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H
#define PVPARALLELVIEW_PVZOOMEDPARALLELVIEWPARAMSWIDGET_H

#include <QToolBar>

#include <pvbase/types.h>

class QStringList;
class QComboBox;

namespace PVParallelView
{
class PVZoomedParallelView;

class PVZoomedParallelViewParamsWidget : public QToolBar
{
	Q_OBJECT

public:
	PVZoomedParallelViewParamsWidget(PVZoomedParallelView* parent);

public:
	void build_axis_menu(int active_col, const QStringList& sl);

signals:
	void change_to_col(int new_axis);

private slots:
	void combo_activated(int index);

private:
	QComboBox *_combo_box;
	PVCol      _active_col;
};

}

#endif // PVPARALLELVIEW_ZOOMEDPARALLELVIEWPARAMSWIDGET_H

