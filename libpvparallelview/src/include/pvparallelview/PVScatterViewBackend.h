/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
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
