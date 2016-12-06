/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H
#define PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H

#include <inendi/PVPlottedNrawCache.h>

#include <pvparallelview/PVScatterViewImagesManager.h>

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
	                     const PVCombCol zone_index,
	                     PVZonesProcessor& zp_bg,
	                     PVZonesProcessor& zp_sel);

	Inendi::PVPlottedNrawCache& get_x_labels_cache() { return _x_labels_cache; }
	Inendi::PVPlottedNrawCache& get_y_labels_cache() { return _y_labels_cache; }

	PVScatterViewImagesManager& get_images_manager() { return _images_manager; }

  private:
	Inendi::PVPlottedNrawCache _x_labels_cache;
	Inendi::PVPlottedNrawCache _y_labels_cache;
	PVScatterViewImagesManager _images_manager;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVSCATTERVIEWBACKEND_H
