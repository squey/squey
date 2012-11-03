#ifndef ZONE_DRAWING_EVENTS_TEST_H
#define ZONE_DRAWING_EVENTS_TEST_H

#include <QMainWindow>

namespace PVParallelView {
class PVLinesView;
}

class LinesViewMw: public QMainWindow
{
	Q_OBJECT

public:
	LinesViewMw() { }

public slots:
	void zr_sel_finished(int zone_id);
	void zr_bg_finished(int zone_id);
};

#endif
