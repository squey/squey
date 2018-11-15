/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <inendi/PVView.h>

#include <pvparallelview/PVScatterViewBackend.h>

/*****************************************************************************
 * PVParallelView::PVScatterViewBackend::PVScatterViewBackend
 *****************************************************************************/

PVParallelView::PVScatterViewBackend::PVScatterViewBackend(
    const Inendi::PVView& view,
    const PVZonesManager& zm,
    PVZonesManager::ZoneRetainer zone_retainer,
    const PVZoneID zone_id,
    PVZonesProcessor& zp_bg,
    PVZonesProcessor& zp_sel)
    : _zone_retainer(std::move(zone_retainer))
    , _x_labels_cache(view, zone_id.first, 100)
    , _y_labels_cache(view, zone_id.second, 100)
    , _images_manager(zone_id,
                      zp_bg,
                      zp_sel,
                      zm,
                      view.get_output_layer_color_buffer(),
                      view.get_real_output_selection())
{
}
