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

PVParallelView::PVScatterViewBackend::PVScatterViewBackend(const Inendi::PVView& view,
                                                           const PVZonesManager& zm,
                                                           const PVCombCol zone_index,
                                                           PVZonesProcessor& zp_bg,
                                                           PVZonesProcessor& zp_sel)
    : _x_labels_cache(view, view.get_axes_combination().get_nraw_axis(zone_index), 100)
    , _y_labels_cache(
          view, view.get_axes_combination().get_nraw_axis(PVCombCol(zone_index + 1)), 100)
    , _images_manager(PVZoneID(zone_index),
                      zp_bg,
                      zp_sel,
                      zm,
                      view.get_output_layer_color_buffer(),
                      view.get_real_output_selection())
{
}
