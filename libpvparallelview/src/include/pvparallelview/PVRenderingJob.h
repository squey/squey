/**
 * \file PVRenderingJob.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVRENDERINGJOB_H
#define PVPARALLELVIEW_PVRENDERINGJOB_H

#include <pvparallelview/common.h>

#include <QObject>

#include <tbb/atomic.h>

namespace PVParallelView {

class PVLinesView;
class PVZoomedParallelScene;
class PVFullParallelScene;

class PVRenderingJob: public QObject
{
	friend class PVLinesView;
	friend class PVZonesManager;
	friend class PVZoomedParallelScene;
	friend class PVFullParallelScene;

	Q_OBJECT

public:
	PVRenderingJob(QObject* parent = NULL):
		QObject(parent)
	{
		_should_cancel = false;
	}

public:
	void cancel() { _should_cancel = true; }
	void reset() { _should_cancel = false; }

protected:
	bool should_cancel() const { return _should_cancel == true; }
	void zone_finished(PVZoneID z) { emit zone_rendered(z); }

signals:
	void zone_rendered(int z);

protected:
	tbb::atomic<bool> _should_cancel;
};

}

#endif