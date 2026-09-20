/* * MIT License
 *
 * © Squey, 2026
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

#ifndef __LINES_VIEW_DROPPED_RENDERINGS_H__
#define __LINES_VIEW_DROPPED_RENDERINGS_H__

#include <QObject>

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneRendering.h>

// Stands for the scene a PVLinesView reports its renderings to, and counts the
// reports.
class RenderingsReceiver : public QObject
{
	Q_OBJECT

  public:
	int reports() const { return _reports; }

  public Q_SLOTS:
	void zr_bg_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);
	void zr_sel_finished(PVParallelView::PVZoneRendering_p zr, PVZoneID zid);

  private:
	int _reports = 0;
};

#endif // __LINES_VIEW_DROPPED_RENDERINGS_H__
