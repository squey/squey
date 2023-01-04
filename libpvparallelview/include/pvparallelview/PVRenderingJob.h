/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
	explicit PVRenderingJob(QObject* parent = nullptr) : QObject(parent) { _should_cancel = false; }

  public:
	void cancel() { _should_cancel = true; }
	void reset() { _should_cancel = false; }

  protected:
	bool should_cancel() const { return _should_cancel == true; }
	void zone_finished(PVZoneID z) { Q_EMIT zone_rendered(z); }

  Q_SIGNALS:
	void zone_rendered(PVZoneID z);

  protected:
	tbb::atomic<bool> _should_cancel;
};
} // namespace PVParallelView

#endif
