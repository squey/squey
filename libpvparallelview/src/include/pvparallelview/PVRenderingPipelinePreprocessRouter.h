/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H
#define PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H

#ifndef PVPARALLELVIEW_PVRENDERING_PIPELINE
#pragma error That file must not be included directly. Use PVRenderingPipeline.h instead
#endif

#include <vector>
#include <list>
#include <tuple>
#include <iostream>

#include <tbb/atomic.h>

namespace PVParallelView {

class PVRenderingPipelinePreprocessRouter
{
	public:
	enum {
		InputIdxDirect = 0,
		InputIdxPostProcess = 1,
	};

	enum {
		OutIdxPreprocess = 0,
		OutIdxContinue = 1,
		OutIdxCancel = 2,
	};

	private:
	enum class ZoneState { NotStarted, Preprocessed };

	/**
	 * Processing state and waiter for a given Zone.
	 */
	struct ZoneInfos
	{
		ZoneInfos() : state(ZoneState::NotStarted)
		{}

		tbb::atomic<ZoneState> state;
		std::list<PVZoneRendering_p> waiters;
	};

	/**
	 * Data for every area.
	 *
	 * * Processing state and colors.
	 */
	struct RouterData
	{
		std::vector<ZoneInfos> _zones_infos;
		PVCore::PVHSVColor const* _colors;
	};

public:
	struct ZoneRenderingWithColors
	{
		// Used by TBB internally
		ZoneRenderingWithColors() { }

		ZoneRenderingWithColors(PVZoneRendering_p zr_, PVCore::PVHSVColor const* colors_):
			zr(zr_), colors(colors_)
		{ }
		
		PVZoneRendering_p zr;
		PVCore::PVHSVColor const* colors;
	};

	using multinode_router = tbb::flow::multifunction_node<PVZoneRendering_p, std::tuple<PVZoneRendering_p, ZoneRenderingWithColors, PVZoneRendering_p> >;

public:
	PVRenderingPipelinePreprocessRouter(size_t nzones, PVCore::PVHSVColor const* colors):
		_d(new RouterData{std::vector<ZoneInfos>{nzones}, colors})
	{}

public:
	void operator()(PVZoneRendering_p zr_in, multinode_router::output_ports_type& op)
	{
		const PVZoneID zone_id = zr_in->get_zone_id();
		assert(zone_id != PVZONEID_INVALID);

		ZoneInfos& infos = _d->_zones_infos[zone_id];
		switch (infos.state) {
			case ZoneState::NotStarted:
				if (!zr_in->should_cancel()) {
					infos.state = ZoneState::Preprocessed;
					std::get<OutIdxPreprocess>(op).try_put(zr_in);
				}
				else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				break;
			case ZoneState::Preprocessed:
			{
				if (!zr_in->should_cancel()) {
					std::get<OutIdxContinue>(op).try_put(ZoneRenderingWithColors(zr_in, _d->_colors));
				}
				else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				break;
			}
		}
	}

public:
	inline void set_zones_count(size_t n) { _d->_zones_infos.resize(n); }
	inline void set_zone_invalid(size_t i) { assert(i < _d->_zones_infos.size()); _d->_zones_infos[i].state = ZoneState::NotStarted; }

private:
	// It is shared_ptr as it is copied by TBB in a multifunction_node while data can be modified
	// from user events.
	std::shared_ptr<RouterData> _d;
};

}

#endif
