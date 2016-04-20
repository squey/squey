/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVZONESPROCESSOR_H
#define PVPARALLELVIEW_PVZONESPROCESSOR_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <tbb/flow_graph.h>

namespace PVCore {
class PVHSVColor;
}

namespace Inendi {
class PVSelection;
}

namespace PVParallelView {

class PVRenderingPipeline;
class PVRenderingPipelinePreprocessRouter;
class PVZonesManager;

/**
 * A ZonesProcessor is a full pipeline with ZonePipeline and preprocessing.
 *
 * It provides interface to recompute or not preprocessing data.
 *
 * @Note it is only sugar for a ZonePileline.
 */
class PVZonesProcessor
{
	using receiver_type = tbb::flow::receiver<PVZoneRendering_p>;

public:
	PVZonesProcessor(receiver_type& in_port, PVRenderingPipelinePreprocessRouter& preprocess):
		_in_port(in_port), _preprocess(preprocess)
	{ }

public:
	/**
	 * Push a new "token" in the pipeline.
	 */
	inline bool add_job(PVZoneRendering_p const& zr) { return _in_port.try_put(zr); }

	/**
	 * Update number of zones.
	 *
	 * @note : Zones to recompute should be invalidate.
	 * @fixme : Every zone should certainly be invalidate in this case.
	 */
	void set_number_zones(const PVZoneID n);

	/**
	 * Invalidate a zone so that it will be recompute from preprocessing.
	 */
	void invalidate_zone_preprocessing(const PVZoneID zone_id);

public:
	/**
	 * Create a ZonesProcessor for foreground image
	 */
	static PVZonesProcessor declare_processor_zm_sel(PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors, Inendi::PVSelection const& sel);

	/**
	 * Create a ZonesProcessor for backgorund image.
	 */
	static PVZonesProcessor declare_background_processor_zm_sel(PVRenderingPipeline& pipeline, PVZonesManager& zm, PVCore::PVHSVColor const* colors, Inendi::PVSelection const& sel);

private:
	receiver_type& _in_port;
	PVRenderingPipelinePreprocessRouter& _preprocess;
};

}

#endif
