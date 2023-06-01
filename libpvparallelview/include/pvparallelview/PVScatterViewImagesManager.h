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

#ifndef __PVSCATTERVIEWIMAGESMANAGER_H__
#define __PVSCATTERVIEWIMAGESMANAGER_H__

#include <boost/utility.hpp>

#include <pvparallelview/common.h>

#include <pvparallelview/PVScatterViewImage.h>
#include <pvparallelview/PVScatterViewData.h>

#include <pvparallelview/PVZoneRenderingScatter_types.h>

namespace PVParallelView
{

class PVZonesManager;
class PVZonesProcessor;

class PVScatterViewImagesManager : boost::noncopyable
{
  protected:
	typedef PVScatterViewData::ProcessParams DataProcessParams;

  public:
	PVScatterViewImagesManager(PVZoneID const zid,
	                           PVZonesProcessor& zp_bg,
	                           PVZonesProcessor& zp_sel,
	                           PVZonesManager const& zm,
	                           const PVCore::PVHSVColor* colors,
	                           Squey::PVSelection const& sel);

	~PVScatterViewImagesManager();

  public:
	bool change_and_process_view(const uint64_t y1_min,
	                             const uint64_t y1_max,
	                             const uint64_t y2_min,
	                             const uint64_t y2_max,
	                             const int zoom,
	                             const double alpha);

  public:
	void process_bg();
	void process_sel();
	void process_all();

  public:
	void set_zone(PVZoneID const zid);
	void cancel_all_and_wait();

  public:
	const QImage& get_image_sel() const;
	const QImage& get_image_all() const;

	PVZoneID get_zone_id() const { return _zid; }
	PVZonesManager const& get_zones_manager() const { return _zm; }

  public:
	inline void set_img_update_receiver(QObject* obj) { _img_update_receiver = obj; }

  private:
	void connect_zr(PVZoneRenderingScatter& zr, const char* slot);

	static void copy_processed_in_processing(DataProcessParams const& params,
	                                         PVScatterViewImage& processing,
	                                         PVScatterViewImage const& processed);

	void process_bg(DataProcessParams const& params);
	void process_sel(DataProcessParams const& params);

  protected:
	PVZoneID _zid;
	PVZonesManager const& _zm;

	PVScatterViewData _data;

	Squey::PVSelection const& _sel;
	PVCore::PVHSVColor const* _colors;

	PVZoneRenderingScatter_p _zr_bg;
	PVZoneRenderingScatter_p _zr_sel;

	PVZonesProcessor& _zp_bg;
	PVZonesProcessor& _zp_sel;

	QObject* _img_update_receiver;
};
} // namespace PVParallelView

#endif // __PVSCATTERVIEWIMAGESMANAGER_H__
