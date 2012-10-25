#ifndef PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H
#define PVPARALLELVIEW_PVRENDERINGPIPELINEPREPROCESSROUTER_H

#ifndef PVPARALLELVIEW_PVRENDERING_PIPELINE
#pragma error That file must not be included directly. Use PVRenderingPipeline.h instead
#endif

#include <vector>
#include <list>
#include <tuple>

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
		ZoneInfos(): state(ZoneStateInvalid) { }

		ZoneState state;
		std::list<PVZoneRenderingBase*> waiters;
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

	typedef tbb::flow::or_node< std::tuple<PVZoneRenderingBase*, PVZoneRenderingBase*> > process_or_type;

protected:
	struct ZoneRenderingWithColors
	{
		// Used by TBB internally
		ZoneRenderingWithColors() { }

		ZoneRenderingWithColors(PVZoneRenderingBase* zr_, PVCore::PVHSVColor* colors_):
			zr(zr_), colors(colors_)
		{ }
		
		PVZoneRenderingBase* zr;
		PVCore::PVHSVColor* colors;
	};

	typedef tbb::flow::multifunction_node<process_or_type::output_type, std::tuple<PVZoneRenderingBase*, ZoneRenderingWithColors, PVZoneRenderingBase*> > multinode_router;

public:
	PVRenderingPipelinePreprocessRouter(size_t nzones, PVCore::PVHSVColor* colors):
		_colors(colors)
	{
		set_zones_count(nzones);
	}

public:
	void operator()(process_or_type::output_type in, multinode_router::output_ports_type& op)
	{
		bool has_been_processed = (in.indx == InputIdxPostProcess);
		PVZoneRenderingBase* r;
		if (has_been_processed) {
			r = std::get<1>(in.result);
		}
		else {
			r = std::get<0>(in.result);
		}

		//std::cout << "Input: " << r.zid << "/" << r.p << ", " << std::get<1>(in) << std::endl;
		const uint32_t zid = r->zid();
		ZoneInfos& infos = zone_infos(zid);
		switch (infos.state) {
			case ZoneStateInvalid:
				if (!r->should_cancel()) {
					infos.state = ZoneStateProcessing;
					std::get<OutIdxPreprocess>(op).try_put(r);
				}
				else {
					std::get<OutIdxCancel>(op).try_put(r);
				}
				break;
			case ZoneStateProcessing:
				if (has_been_processed) {
					infos.state = ZoneStateValid;
				}
				else {
					if (!r->should_cancel()) {
						infos.waiters.push_back(r);
					}
					else {
						std::get<OutIdxCancel>(op).try_put(r);
					}
					break;
				}
			case ZoneStateValid:
			{
				// Put everyone and waiters
				if (!r->should_cancel()) {
					std::get<OutIdxContinue>(op).try_put(ZoneRenderingWithColors(r, _colors));
				}
				else {
					std::get<OutIdxCancel>(op).try_put(r);
				}
				for (PVZoneRenderingBase* zr: infos.waiters) {
					if (!zr->should_cancel()) {
						std::get<OutIdxContinue>(op).try_put(ZoneRenderingWithColors(zr, _colors));
					}
					else {
						std::get<OutIdxCancel>(op).try_put(r);
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
		_zones_infos.resize(n);
	}

	inline ZoneState zone_state(size_t i) const { assert(i < _zones_infos.size()); return _zones_infos[i].state; }
	inline void set_zone_valid(size_t i) { assert(i < _zones_infos.size()); _zones_infos[i].state = ZoneStateValid; }

private:
	inline ZoneInfos& zone_infos(size_t i) { assert(i < _zones_infos.size()); return _zones_infos[i]; }
	inline ZoneInfos const& zone_infos(size_t i) const { assert(i < _zones_infos.size()); return _zones_infos[i]; }

private:
	std::vector<ZoneInfos> _zones_infos;
	PVCore::PVHSVColor* _colors;
};

}

#endif
