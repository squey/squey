//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

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
