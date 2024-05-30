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

#ifndef PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H
#define PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H

#include <pvparallelview/PVAbstractAxisSliders.h>
#include <pvparallelview/PVAbstractAxisSlider.h>

class QGraphicsItem;

namespace PVParallelView
{

class PVSlidersGroup;

class PVAbstractRangeAxisSliders : public PVAbstractAxisSliders
{
  public:
	typedef PVSlidersManager::id_t id_t;
	typedef std::pair<int64_t, int64_t> range_t;

  public:
	PVAbstractRangeAxisSliders(QGraphicsItem* parent,
	                           PVSlidersManager* sm_p,
	                           PVSlidersGroup* group,
	                           const char* text);

	~PVAbstractRangeAxisSliders() override;

	// FIXME : This is an Ugly interface with a lot of bad use possibility.
	virtual void initialize(id_t id, int64_t y_min, int64_t y_max) = 0;

	bool is_moving() const override
	{
		return (_sl_min and _sl_max) ? (_sl_min->is_moving() || _sl_max->is_moving()) : false;
	}

	range_t get_range() const
	{
		if (not _sl_min or not _sl_max) {
			return range_t{0, 0};
		}
		int64_t v_min = _sl_min->get_value();
		int64_t v_max = _sl_max->get_value();

		return std::make_pair(std::min(v_min, v_max), std::max(v_min, v_max));
	}

	void refresh() override
	{
		if (_sl_min and _sl_max) {
			refresh_value(_sl_min->get_value(), _sl_max->get_value());
		}
	}

  protected:
	void refresh_value(int64_t y_min, int64_t y_max)
	{
		_sl_min->set_value(y_min);
		_sl_max->set_value(y_max);
	}

  protected:
	void paint(QPainter* painter,
	           const QStyleOptionGraphicsItem* option,
	           QWidget* widget = nullptr) override;

  protected:
	PVAbstractAxisSlider* _sl_min;
	PVAbstractAxisSlider* _sl_max;
};
} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVABSTRACTRANGEAXISSLIDERS_H
