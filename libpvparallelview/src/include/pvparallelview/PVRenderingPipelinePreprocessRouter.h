/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H
#define PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H

#ifndef PVPARALLELVIEW_PVRENDERING_PIPELINE
#pragma error That file must not be included directly.Use PVRenderingPipeline.h instead
#endif

#include <vector>
#include <tuple>

#include <tbb/atomic.h>

namespace PVParallelView
{

/**
 * This class is a functor to fill the pipeline applying preprocessing before if required.
 *
 * @note: It handles ZoneRending cancellation.
 */
class PVRenderingPipelinePreprocessRouter
{
  public:
	/**
	 * Identifier for output of this function router.
	 */
	enum {
		OutIdxPreprocess = 0,
		OutIdxContinue = 1,
		OutIdxCancel = 2,
	};

	/**
	 * Gather all data required for processing from the pipeline.
	 */
	struct ZoneRenderingWithColors {
		// Used by TBB internally
		ZoneRenderingWithColors() {}

		ZoneRenderingWithColors(PVZoneRendering_p zr_, PVCore::PVHSVColor const* colors_)
		    : zr(std::move(zr_)), colors(colors_)
		{
		}

		PVZoneRendering_p zr;
		PVCore::PVHSVColor const* colors;
	};

	using multinode_router = tbb::flow::multifunction_node<
	    PVZoneRendering_p,
	    std::tuple<PVZoneRendering_p, ZoneRenderingWithColors, PVZoneRendering_p>>;

  public:
	PVRenderingPipelinePreprocessRouter(PVZonesManager const& zm, PVCore::PVHSVColor const* colors)
	    : _d(new RouterData{
	          std::vector<ZoneInfos>{zm.get_number_of_zones(), {ZoneState::NotStarted, {}}}, colors,
	          zm})
	{
	}

  public:
	/**
	 * Routing function for ZoneRendering input.
	 *
	 * Normal path is : Preprocess ZoneRendering then push it in the pipeline.
	 * Preprocessing can be skipped if it is already done.
	 * ZoneRendering Processing can be canceled.
	 *
	 * @warning : Cancelling possibility is certainly a loss of processing time.
	 */
	void operator()(PVZoneRendering_p zr_in, multinode_router::output_ports_type& op)
	{
		const PVZoneID zone_id = zr_in->get_zone_id();
		assert(zone_id != PVZONEID_INVALID);
		assert(_d->_zone_manager.has_zone(zone_id));

		for (auto& zone_index : _d->_zone_manager.get_zone_indices(zone_id)) {
			ZoneInfos& infos = _d->_zones_infos[zone_index];
			switch (infos.state) {
			case ZoneState::NotStarted:
				if (!zr_in->should_cancel()) {
					infos.state = ZoneState::Processing;
					std::get<OutIdxPreprocess>(op).try_put(zr_in);
				} else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				break;
			case ZoneState::Processing:
				// It may happen that parallel view is currently processing B code while scatter
				// required it too. Scatter subscribe in waiters list so that once parallel view is
				// reinserted in the router, it will launcher scatter too in the pipeline.
				infos.waiters.push_back(zr_in);
				break;
			case ZoneState::Preprocessed: {
				if (!zr_in->should_cancel()) {
					std::get<OutIdxContinue>(op).try_put(
					    ZoneRenderingWithColors(zr_in, _d->_colors));
				} else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				for (auto& zr : infos.waiters) {
					if (!zr->should_cancel()) {
						std::get<OutIdxContinue>(op).try_put(
						    ZoneRenderingWithColors(zr, _d->_colors));
					} else {
						std::get<OutIdxCancel>(op).try_put(zr);
					}
				}
				infos.waiters.clear();
				break;
			}
			}
		}
	}

	// Preprocessing function should inform the pipeline that it is done.
	void preprocessing_done(PVZoneID zid)
	{
		assert(_d->_zone_manager.has_zone(zid));
		for (auto& zone_index : _d->_zone_manager.get_zone_indices(zid)) {
			_d->_zones_infos[zone_index].state = ZoneState::Preprocessed;
		}
	}

  public:
	/**
	 * Interface to update pipeline information for future processing.
	 */
	inline void set_zone_invalid(PVZoneID zid)
	{
		assert(_d->_zone_manager.has_zone(zid));
		for (auto& zone_index : _d->_zone_manager.get_zone_indices(zid)) {
			_d->_zones_infos[zone_index].state = ZoneState::NotStarted;
		}
	}
	void reset_zones_count(size_t n)
	{
		_d->_zones_infos.resize(n);
		for (auto& zinfo : _d->_zones_infos) {
			zinfo.state = ZoneState::NotStarted;
		}
	}

  private:
	/**
	 * Preprocessing possible state for zone rendering.
	 */
	enum class ZoneState { NotStarted, Preprocessed, Processing };

	/**
	 * Processing state for a given Zone.
	 */
	struct ZoneInfos {
		tbb::atomic<ZoneState> state;
		std::vector<PVZoneRendering_p> waiters;
	};

	/**
	 * Data for every zone.
	 *
	 * * Processing state and colors.
	 */
	struct RouterData {
		std::vector<ZoneInfos> _zones_infos;
		PVCore::PVHSVColor const* _colors;
		PVZonesManager const& _zone_manager;
	};

  private:
	// It is shared_ptr as it is copied by TBB in a multifunction_node while data can be modified
	// from user events.
	std::shared_ptr<RouterData> _d;
};
} // namespace PVParallelView

#endif
