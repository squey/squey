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

#ifndef PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
#define PVPARALLELVIEW_PVZOOMAXISSLIDERS_H

#include <sigc++/sigc++.h>

#include <pvkernel/core/PVAlgorithms.h>

#include <pvparallelview/PVAbstractRangeAxisSliders.h>
#include <pvparallelview/PVSlidersManager.h>

namespace PVParallelView
{

class PVSlidersGroup;

class PVZoomAxisSliders : public PVAbstractRangeAxisSliders, public sigc::trackable
{
	Q_OBJECT

  private:
	typedef PVSlidersManager::id_t id_t;

  public:
	PVZoomAxisSliders(QGraphicsItem* parent, PVSlidersManager* sm_p, PVSlidersGroup* group);

	void initialize(id_t id, int64_t y_min, int64_t y_max) override;

  public Q_SLOTS:
	void remove_from_axis() override;

  private Q_SLOTS:
	void do_sliders_moved();

  private:
	void on_zoom_sliders_update(PVCombCol col,
	                            id_t id,
	                            int64_t y_min,
	                            int64_t y_max,
	                            PVSlidersManager::ZoomSliderChange change);

  private:
	id_t _id;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVZOOMAXISSLIDERS_H
