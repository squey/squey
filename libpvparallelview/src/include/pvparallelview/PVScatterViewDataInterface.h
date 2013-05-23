/**
 * \file PVScatterViewDataInterface.h
 *
 * Copyright (C) Picviz Labs 2013
 */

#ifndef __PVSCATTERVIEWDATAINTERFACE_H__
#define __PVSCATTERVIEWDATAINTERFACE_H__


#include <boost/noncopyable.hpp>

#include <pvkernel/core/general.h>
#include <pvkernel/core/PVHSVColor.h>

#include <pvparallelview/PVScatterViewImage.h>

namespace tbb {
class task_group_context;
}

namespace Picviz {
class PVSelection;
}

namespace PVParallelView {

class PVZoomedZoneTree;

class PVScatterViewDataInterface : boost::noncopyable
{
public:
	PVScatterViewDataInterface() {};
	virtual ~PVScatterViewDataInterface() {};

public:
	struct ProcessParams
	{
		struct dirty_rect
		{
			dirty_rect() : y1_min(0), y1_max(0), y2_min(0), y2_max(0) {}
			uint64_t y1_min;
			uint64_t y1_max;
			uint64_t y2_min;
			uint64_t y2_max;
		};

		ProcessParams(
			PVZoomedZoneTree const& zzt_,
			const PVCore::PVHSVColor* colors_
		) :
			zzt(&zzt_),
			colors(colors_),
			y1_min(0),
			y1_max(0),
			y2_min(0),
			y2_max(0),
			zoom(0),
			alpha(1.0),
			y1_offset(0),
			y2_offset(0)
		{ }

		ProcessParams():
			zzt(nullptr),
			colors(nullptr),
			y1_min(-1),
			y1_max(-1),
			y2_min(-1),
			y2_max(-1),
			zoom(-1),
			alpha(1.0),
			y1_offset(0),
			y2_offset(0)
		{ }

		dirty_rect rect_1() const;
		dirty_rect rect_2() const;
		int32_t map_to_view(int64_t scene_value) const;
		QRect map_to_view(const dirty_rect& rect) const;

		inline bool params_changed(
				uint64_t y1_min_,
				uint64_t y1_max_,
				uint64_t y2_min_,
				uint64_t y2_max_,
				int zoom_,
				double alpha_) const
		{
			return !(y1_min_ == y1_min &&
					y1_max_ == y1_max &&
					y2_min_ == y2_min &&
					y2_max_ == y2_max &&
					zoom_ == zoom &&
					alpha_ == alpha);
		}

		void set_params(
				uint64_t y1_min_,
				uint64_t y1_max_,
				uint64_t y2_min_,
				uint64_t y2_max_,
				int zoom_,
				double alpha_
		)
		{
			// Translation
			if (zoom_ == zoom && alpha_ == alpha) {
				y1_offset = y1_min - y1_min_;
				y2_offset = y2_min - y2_min_;
			}
			else {
				y1_offset = 0;
				y2_offset = 0;
			}
			y1_min = y1_min_;
			y1_max = y1_max_;
			y2_min = y2_min_;
			y2_max = y2_max_;
			zoom = zoom_;
			alpha = alpha_;
		}

		PVZoomedZoneTree const* zzt;
		const PVCore::PVHSVColor* colors;
		uint64_t y1_min;
		uint64_t y1_max;
		uint64_t y2_min;
		uint64_t y2_max;
		int zoom;
		double alpha;
		int64_t y1_offset;
		int64_t y2_offset;
	};

	class ProcessImage
	{
	public:
		inline PVScatterViewImage& image_processing() { return _image_processing; }
		inline PVScatterViewImage& image_processed()  { return _image_processed; }
		inline PVScatterViewImage const& image_processed() const { return _image_processed; }
		inline ProcessParams const& processed_params() const { return _processed_params; }

		void swap(ProcessParams const& processed_params)
		{
			_image_processed.swap(_image_processing);
			_processed_params = processed_params;
			_processed_params.y1_offset = 0;
			_processed_params.y2_offset = 0;
		}

	private:
		ProcessParams _processed_params;
		PVScatterViewImage _image_processed;
		PVScatterViewImage _image_processing;
	};

public:
	inline void process_image_bg(ProcessParams const& params, tbb::task_group_context* ctxt = nullptr)
	{
		process_bg(params, image_bg_processing(), ctxt);
		if (!is_ctxt_cancelled(ctxt)) {
			_image_bg.swap(params);
		}
	}

	inline void process_image_sel(ProcessParams const& params, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt = nullptr)
	{
		process_sel(params, image_sel_processing(), sel, ctxt);
		if (!is_ctxt_cancelled(ctxt)) {
			_image_sel.swap(params);
		}
	}

	inline void process_all_images(ProcessParams const& params, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt = nullptr)
	{
		process_all(params, image_bg_processing(), image_sel_processing(), sel, ctxt);
		if (!is_ctxt_cancelled(ctxt)) {
			_image_bg.swap(params);
			_image_sel.swap(params);
		}
	}

public:
	PVScatterViewImage const& image_bg() const { return _image_bg.image_processed(); }
	PVScatterViewImage const& image_sel() const { return _image_sel.image_processed(); }

	PVScatterViewImage& image_bg() { return _image_bg.image_processed(); }
	PVScatterViewImage& image_sel() { return _image_sel.image_processed(); }

	PVScatterViewImage& image_bg_processing() { return _image_bg.image_processing(); }
	PVScatterViewImage& image_sel_processing() { return _image_sel.image_processing(); }

	ProcessParams const& image_bg_process_params() { return _image_bg.processed_params(); }
	ProcessParams const& image_sel_process_params() { return _image_sel.processed_params(); }


	void clear_processing()
	{
		image_bg_processing().clear();
		image_sel_processing().clear();
	}

protected:
	virtual void process_bg(ProcessParams const& params, PVScatterViewImage& image, tbb::task_group_context* ctxt = nullptr) const = 0;
	virtual void process_sel(ProcessParams const& params, PVScatterViewImage& image, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt = nullptr) const = 0;
	virtual void process_all(ProcessParams const& params, PVScatterViewImage& image_bg, PVScatterViewImage& image_sel, Picviz::PVSelection const& sel, tbb::task_group_context* ctxt = nullptr) const
	{
		process_bg(params, image_bg, ctxt);
		if (!is_ctxt_cancelled(ctxt)) {
			process_sel(params, image_sel, sel, ctxt);
		}
	}

protected:
	static bool is_ctxt_cancelled(tbb::task_group_context* ctxt);


private:
	ProcessImage _image_bg;
	ProcessImage _image_sel;
};

}


#endif // __PVSCATTERVIEWDATAINTERFACE_H__
