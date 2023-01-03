/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVPARALLELVIEW_PVZONESPROCESSOR_H
#define PVPARALLELVIEW_PVZONESPROCESSOR_H

#include <pvparallelview/common.h>
#include <pvparallelview/PVZoneRendering_types.h>

#include <tbb/flow_graph.h>

namespace PVCore
{
class PVHSVColor;
} // namespace PVCore

namespace Inendi
{
class PVSelection;
} // namespace Inendi

namespace PVParallelView
{

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
	PVZonesProcessor(receiver_type& in_port, PVRenderingPipelinePreprocessRouter& preprocess)
	    : _in_port(in_port), _preprocess(preprocess)
	{
	}

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
	void reset_number_zones(const size_t n);

	/**
	 * Invalidate a zone so that it will be recompute from preprocessing.
	 */
	void invalidate_zone_preprocessing(const PVZoneID zone_id);

  public:
	/**
	 * Create a ZonesProcessor for foreground image
	 */
	static PVZonesProcessor declare_processor_zm_sel(PVRenderingPipeline& pipeline,
	                                                 PVZonesManager& zm,
	                                                 PVCore::PVHSVColor const* colors,
	                                                 Inendi::PVSelection const& sel);

	/**
	 * Create a ZonesProcessor for background image.
	 */
	static PVZonesProcessor declare_background_processor_zm_sel(PVRenderingPipeline& pipeline,
	                                                            PVZonesManager& zm,
	                                                            PVCore::PVHSVColor const* colors,
	                                                            Inendi::PVSelection const& sel);

  private:
	receiver_type& _in_port;
	PVRenderingPipelinePreprocessRouter& _preprocess;
};
} // namespace PVParallelView

#endif
