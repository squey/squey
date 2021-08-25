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

#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZonesManager.h>

/******************************************************************************
 *
 * PVParallelView::PVZonesProcessor::declare_processor_zm_sel
 *
 *****************************************************************************/
PVParallelView::PVZonesProcessor
PVParallelView::PVZonesProcessor::declare_processor_zm_sel(PVRenderingPipeline& pipeline,
                                                           PVZonesManager& zm,
                                                           PVCore::PVHSVColor const* colors,
                                                           Inendi::PVSelection const& sel)
{
	return pipeline.declare_processor(
	    [&](PVZoneID zone_id) { zm.filter_zone_by_sel(zone_id, sel); }, colors, zm);
}

PVParallelView::PVZonesProcessor
PVParallelView::PVZonesProcessor::declare_background_processor_zm_sel(
    PVRenderingPipeline& pipeline,
    PVZonesManager& zm,
    PVCore::PVHSVColor const* colors,
    Inendi::PVSelection const& sel)
{
	return pipeline.declare_processor(
	    [&](PVZoneID zone_id) { zm.filter_zone_by_sel_background(zone_id, sel); }, colors, zm);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesProcessor::invalidate_zone_preprocessing
 *
 *****************************************************************************/
void PVParallelView::PVZonesProcessor::invalidate_zone_preprocessing(const PVZoneID zone_id)
{
	_preprocess.set_zone_invalid(zone_id);
}

/******************************************************************************
 *
 * PVParallelView::PVZonesProcessor::reset_number_zones
 *
 *****************************************************************************/
void PVParallelView::PVZonesProcessor::reset_number_zones(const size_t n)
{
	_preprocess.reset_zones_count(n);
}
