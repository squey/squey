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

#define ROUTER_INPUT_IDX_DIRECT 0
#define ROUTER_INPUT_IDX_POSTPROCESS 1

namespace PVParallelView {

class PVRenderingPipeline;

class PVRenderingPipelinePreprocessRouter
{
	friend class PVRenderingPipeline;

	typedef enum {
		ZoneStateInvalid = 0,
		ZoneStateValid,
		ZoneStateProcessing
	} ZoneState;

	struct ZoneInfos
	{
		ZoneInfos()
		{
			state = ZoneStateInvalid;
		}

		tbb::atomic<ZoneState> state;
		std::list<PVZoneRenderingBase_p> waiters;
	};

	enum {
		InputIdxDirect = 0,
		InputIdxPostProcess = 1,
	};

	enum {
		OutIdxPreprocess = 0,
		OutIdxContinue = 1,
		OutIdxCancel = 2,
	};

	typedef tbb::flow::or_node< std::tuple<PVZoneRenderingBase_p, PVZoneRenderingBase_p> > process_or_type;

	struct RouterData
	{
		std::vector<ZoneInfos> _zones_infos;
		PVCore::PVHSVColor const* _colors;
	};

protected:
	struct ZoneRenderingWithColors
	{
		// Used by TBB internally
		ZoneRenderingWithColors() { }

		ZoneRenderingWithColors(PVZoneRenderingBase_p zr_, PVCore::PVHSVColor const* colors_):
			zr(zr_), colors(colors_)
		{ }
		
		PVZoneRenderingBase_p zr;
		PVCore::PVHSVColor const* colors;
	};

	typedef tbb::flow::multifunction_node<process_or_type::output_type, std::tuple<PVZoneRenderingBase_p, ZoneRenderingWithColors, PVZoneRenderingBase_p> > multinode_router;

public:
	PVRenderingPipelinePreprocessRouter(size_t nzones, PVCore::PVHSVColor const* colors):
		_d(new RouterData())
	{
		_d->_colors = colors;
		set_zones_count(nzones);
	}

public:
	void operator()(process_or_type::output_type in, multinode_router::output_ports_type& op)
	{
		bool has_been_processed = (in.indx == InputIdxPostProcess);
		PVZoneRenderingBase_p zr_in;
		if (has_been_processed) {
			zr_in = std::get<1>(in.result);
		}
		else {
			zr_in = std::get<0>(in.result);
		}

		const uint32_t zone_id = zr_in->get_zone_id();
		ZoneInfos& infos = zone_infos(zone_id);
		switch (infos.state) {
			case ZoneStateInvalid:
				if (!zr_in->should_cancel()) {
					infos.state = ZoneStateProcessing;
					std::get<OutIdxPreprocess>(op).try_put(zr_in);
				}
				else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				break;
			case ZoneStateProcessing:
				if (has_been_processed) {
					infos.state = ZoneStateValid;
				}
				else {
					if (!zr_in->should_cancel()) {
						infos.waiters.push_back(zr_in);
					}
					else {
						std::get<OutIdxCancel>(op).try_put(zr_in);
					}
					break;
				}
			case ZoneStateValid:
			{
				// Put everyone and waiters
				if (!zr_in->should_cancel()) {
					std::get<OutIdxContinue>(op).try_put(ZoneRenderingWithColors(zr_in, _d->_colors));
				}
				else {
					std::get<OutIdxCancel>(op).try_put(zr_in);
				}
				for (PVZoneRenderingBase_p const& zr_wait: infos.waiters) {
					if (!zr_wait->should_cancel()) {
						std::get<OutIdxContinue>(op).try_put(ZoneRenderingWithColors(zr_wait, _d->_colors));
					}
					else {
						std::get<OutIdxCancel>(op).try_put(zr_wait);
					}
				}
				infos.waiters.clear();
				break;
			}
		}
	}

public:
	inline void set_zones_count(size_t n)
	{
		_d->_zones_infos.resize(n);
	}

	inline ZoneState zone_state(size_t i) const { assert(i < _d->_zones_infos.size()); return _d->_zones_infos[i].state; }
	inline void set_zone_invalid(size_t i) { assert(i < _d->_zones_infos.size()); _d->_zones_infos[i].state = ZoneStateInvalid; }

private:
	inline ZoneInfos& zone_infos(size_t i) { assert(i < _d->_zones_infos.size()); return _d->_zones_infos[i]; }
	inline ZoneInfos const& zone_infos(size_t i) const { assert(i < _d->_zones_infos.size()); return _d->_zones_infos[i]; }

private:
	boost::shared_ptr<RouterData> _d;
};

}

#endif
