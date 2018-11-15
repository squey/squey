/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

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
