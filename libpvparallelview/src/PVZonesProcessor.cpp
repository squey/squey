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
PVParallelView::PVZonesProcessor PVParallelView::PVZonesProcessor::declare_processor_zm_sel(
    PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors,
    Inendi::PVSelection const& sel)
{
	return pipeline.declare_processor(
	    [&](PVZoneID zone_id) { zm.filter_zone_by_sel(zone_id, sel); }, colors,
	    zm.get_number_of_managed_zones());
}

PVParallelView::PVZonesProcessor
PVParallelView::PVZonesProcessor::declare_background_processor_zm_sel(
    PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors,
    Inendi::PVSelection const& sel)
{
	return pipeline.declare_processor(
	    [&](PVZoneID zone_id) { zm.filter_zone_by_sel_background(zone_id, sel); }, colors,
	    zm.get_number_of_managed_zones());
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
 * PVParallelView::PVZonesProcessor::set_number_zones
 *
 *****************************************************************************/
void PVParallelView::PVZonesProcessor::set_number_zones(const PVZoneID zone_id)
{
	_preprocess.set_zones_count(zone_id);
}
