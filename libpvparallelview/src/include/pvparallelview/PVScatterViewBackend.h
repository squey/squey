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

#ifndef PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H
#define PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H

#include <inendi/PVPlottedNrawCache.h>

#include <pvparallelview/PVScatterViewImagesManager.h>
#include <pvparallelview/PVZonesManager.h>

namespace Inendi
{
class PVView;
class PVPlottedNrawCache;
} // namespace Inendi

namespace PVParallelView
{

class PVZonesManager;
class PVZonesProcessor;

class PVScatterViewBackend
{
  public:
	PVScatterViewBackend(const Inendi::PVView& view,
	                     const PVZonesManager& zm,
	                     PVZonesManager::ZoneRetainer zone_retainer,
	                     const PVZoneID zone_id,
	                     PVZonesProcessor& zp_bg,
	                     PVZonesProcessor& zp_sel);

	Inendi::PVPlottedNrawCache& get_x_labels_cache() { return _x_labels_cache; }
	Inendi::PVPlottedNrawCache& get_y_labels_cache() { return _y_labels_cache; }

	PVScatterViewImagesManager& get_images_manager() { return _images_manager; }

  private:
	PVZonesManager::ZoneRetainer _zone_retainer;
	Inendi::PVPlottedNrawCache _x_labels_cache;
	Inendi::PVPlottedNrawCache _y_labels_cache;
	PVScatterViewImagesManager _images_manager;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H
