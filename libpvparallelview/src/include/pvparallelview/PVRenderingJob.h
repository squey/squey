/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVRENDERINGJOB_H
#define PVPARALLELVIEW_PVRENDERINGJOB_H

#include <pvparallelview/common.h>

#include <QObject>

#include <tbb/atomic.h>

namespace PVParallelView
{

class PVLinesView;
class PVZoomedParallelScene;
class PVFullParallelScene;

class PVRenderingJob : public QObject
{
	friend class PVLinesView;
	friend class PVZonesManager;
	friend class PVZoomedParallelScene;
	friend class PVFullParallelScene;

	Q_OBJECT

  public:
	PVRenderingJob(QObject* parent = nullptr) : QObject(parent) { _should_cancel = false; }

  public:
	void cancel() { _should_cancel = true; }
	void reset() { _should_cancel = false; }

  protected:
	bool should_cancel() const { return _should_cancel == true; }
	void zone_finished(PVZoneID z) { Q_EMIT zone_rendered(z); }

  Q_SIGNALS:
	void zone_rendered(int z);

  protected:
	tbb::atomic<bool> _should_cancel;
};
}

#endif
