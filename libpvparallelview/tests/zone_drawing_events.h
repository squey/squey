/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
