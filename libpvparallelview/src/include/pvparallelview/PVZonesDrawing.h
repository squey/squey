/**
 * \file PVZonesDrawing.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONESDRAWING_H
#define PVPARALLELVIEW_PVZONESDRAWING_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
#include <pvparallelview/PVBCIBuffers.h>
#include <pvparallelview/PVBCICode.h>
#include <pvparallelview/PVBCIDrawingBackend.h>
#include <pvparallelview/PVBCIBackendImage_types.h>
#include <pvparallelview/PVZonesManager.h>
#include <pvparallelview/PVBCIBackendImage.h>

#include <boost/array.hpp>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/utility.hpp>

#include <tbb/enumerable_thread_specific.h>

#include <QFuture>
#include <qtconcurrentmap.h>

#include <cassert>
#include <functional>


namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

class PVBCIDrawingBackend;

namespace __impl {

struct PVZonesDrawingBase
{
	// BCI buffers are shared among all instances of PVZonesDrawing
	static PVBCIBuffers<BCI_BUFFERS_COUNT> _computed_codes;
};

}

template <size_t Bbits>
class PVZonesDrawing: boost::noncopyable, public __impl::PVZonesDrawingBase
{
	typedef boost::array<PVBCICodeBase, NBUCKETS> array_codes_t;
	typedef tbb::enumerable_thread_specific<array_codes_t> tls_codes_t;

public:
	typedef PVBCIDrawingBackend bci_backend_t;
	typedef typename bci_backend_t::backend_image_t backend_image_t;
	typedef typename bci_backend_t::backend_image_p_t backend_image_p_t;
	typedef PVBCICodeBase bci_codes_t;
	typedef PVZoomedZoneTree::context_t zzt_context_t;

public:
	PVZonesDrawing(PVZonesManager& zm, bci_backend_t& backend, PVCore::PVHSVColor const& colors):
		_zm(zm),
		_draw_backend(&backend),
		_colors(&colors)
	{ }

	~PVZonesDrawing()
	{ }

/*public:
	template <typename Tree, typename Fbci>
	void draw(QImage& dst_img, Tree const& tree, Fbci const& f_bci);*/

public:
	inline void set_backend(bci_backend_t const& backend) { _draw_backend = &backend; }

public:
	inline PVBCIBackendImage_p create_image(size_t width) const
	{
		assert(_draw_backend);
		PVBCIBackendImage_p ret = _draw_backend->create_image(width, Bbits);
		ret->set_width(width);
		return ret;
	}

public:
#if 0
	template <class Fbci, class BackendImageIterator>
	void draw_zones(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			draw_zone<Fbci>(dst_img_begin, 0, zone, f_bci);
			dst_img_begin++;
		}
	}

	template <class Fbci, class BackendImageIterator>
	inline QFuture<void> draw_zones_futur(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		return draw_zones_futur_lambda(dst_img_begin, zone_start, nzones,
			[&](PVZoneTree const& zone_tree, PVCore::PVHSVColor const* colors, PVBCICode<Bbits>* codes)
			{
				return (zone_tree.*f_bci)(colors, codes);
			}
	   );
	}

	template <class Fbci, class BackendImageIterator>
	QFuture<void> draw_zones_futur_lambda(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		return QtConcurrent::map(boost::counting_iterator<PVZoneID>(zone_start), boost::counting_iterator<PVZoneID>(nzones),
			[&](PVZoneID zone)
			{
				// Get free BCI buffer
				PVBCICode<Bbits>* codes = _computed_codes.get_available_buffer<Bbits>();

				// Get the BCI codes
				PVZoneTree const& zone_tree = _zm.get_zone_tree<PVZoneTree>(zone);
				size_t ncodes = f_bci(zone_tree, _colors, codes);

				// And draw them...
				PVBCIBackendImage<Bbits> &dst_img = *(*(dst_img_begin + zone));
				draw_bci(dst_img, 0, dst_img.width(), codes, ncodes);

				_computed_codes.return_buffer<Bbits>(codes);
			}
		);
	}

	template <class Fbci>
	uint32_t draw_zones(PVBCIBackendImage<Bbits>& dst_img, uint32_t x_start, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			//assert(x_start + _zm.get_zone_width(zone) + AxisWidth <= dst_img.width());
			draw_zone<Fbci>(dst_img, x_start, zone, f_bci);
			x_start += _zm.get_zone_width(zone) + AxisWidth;
		}
		return x_start;
	}
#endif

	template <class Fbci>
	inline void draw_zone(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, uint32_t zone_width, Fbci const& f_bci, const float zoom_y = 1.0f)
	{
		PVZoneTree const &zone_tree = _zm.get_zone_tree<PVZoneTree>(zone);
		draw_bci_lambda<PVZoneTree>(zone_tree, dst_img, x_start, zone_width,
				[&](PVZoneTree const& zone_tree, PVCore::PVHSVColor const* colors, PVBCICode<Bbits>* codes)
				{
					return (zone_tree.*f_bci)(colors, codes);
				},
				zoom_y,
				false
			);
	}


	template <class Fbci>
	inline void draw_zoomed_zone(zzt_context_t &ctx, PVBCIBackendImage &dst_img, uint64_t y_min, uint64_t y_max, uint64_t y_lim, int zoom, PVZoneID zone, Fbci const &f_bci, const float zoom_y = 1.0f, const float zoom_x = 1.0f, bool reverse = false)
	{
		PVZoomedZoneTree const &zoomed_zone_tree = _zm.get_zone_tree<PVZoomedZoneTree>(zone);
		draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
			(zoomed_zone_tree, dst_img, 0, dst_img.width(),
			 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
			     PVCore::PVHSVColor const* colors,
			     PVParallelView::PVBCICode<Bbits>* codes)
			 {
				 return (zoomed_zone_tree.*f_bci)(ctx, y_min, y_max, y_lim, zoom,
				                                  dst_img.width(), colors,
				                                  codes, zoom_x);
			 }, zoom_y, reverse);
	}

	template <class Fbci>
	inline void draw_zoomed_zone_sel(zzt_context_t &ctx, PVBCIBackendImage &dst_img, uint64_t y_min, uint64_t y_max, uint64_t y_lim, Picviz::PVSelection &selection, int zoom, PVZoneID zone, Fbci const &f_bci, const float zoom_y = 1.0f, const float zoom_x = 1.0f, bool reverse = false)
	{
		PVZoomedZoneTree const &zoomed_zone_tree = _zm.get_zone_tree<PVZoomedZoneTree>(zone);
		draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
			(zoomed_zone_tree, dst_img, 0, dst_img.width(),
			 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
			     PVCore::PVHSVColor const* colors,
			     PVParallelView::PVBCICode<Bbits>* codes)
			 {
				 return (zoomed_zone_tree.*f_bci)(ctx, y_min, y_max, y_lim, selection,
				                                  zoom, dst_img.width(), colors,
				                                  codes, zoom_x);
			 }, zoom_y, reverse);
	}


	template <class Tree, class Fbci>
	void draw_bci_lambda(Tree const &zone_tree, backend_image_t& dst_img, uint32_t x_start, size_t width, Fbci const& f_bci, const float zoom_y, bool reverse = false)
	{
		// Get a free BCI buffers
		PVBCICode<Bbits>* bci_buf = &_computed_codes.get_available_buffer()->as<Bbits>();
		size_t ncodes = f_bci(zone_tree, _colors, bci_buf);
		draw_bci(dst_img, x_start, width, bci_buf, ncodes, zoom_y, reverse,
				[=]
				{
					PVZonesDrawingBase::_computed_codes.return_buffer(reinterpret_cast<PVBCICodeBase*>(bci_buf));
				}, zoom_y, reverse);
	}

	inline const PVZonesManager&  get_zones_manager() const
	{
		return _zm;
	}

	inline PVZonesManager&  get_zones_manager()
	{
		return _zm;
	}

private:
	void draw_bci(backend_image_p_t& dst_img, uint32_t x_start, size_t width, bci_codes_t* codes, size_t n, const float zoom_y = 1.0f, bool reverse = false)
	{
		(*_draw_backend)(dst_img, x_start, width, codes, n, zoom_y, reverse);
	}

private:
	PVZonesManager& _zm;
	bci_backend_t* _draw_backend;
	PVCore::PVHSVColor const* _colors;
};

}

#endif
