#include <pvparallelview/PVRenderingPipeline.h>
#include <pvparallelview/PVZonesManager.h>

PVParallelView::PVZonesProcessor PVParallelView::PVZonesProcessor::declare_processor_zm_sel(PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors, Picviz::PVSelection const& sel)
{
	return pipeline.declare_processor(
		[&](PVZoneID z)
		{
			zm.filter_zone_by_sel(z, sel);
		},
		colors, zm.get_number_zones());
}

PVParallelView::PVZonesProcessor PVParallelView::PVZonesProcessor::declare_processor_direct(PVRenderingPipeline& pipeline, PVCore::PVHSVColor const* colors)
{
	return pipeline.declare_processor(colors);
}

void PVParallelView::PVZonesProcessor::invalidate_zone_preprocessing(const PVZoneID z)
{
	if (_preprocess) {
		_preprocess->set_zone_invalid(z);
	}
}
