/**
 * \file PVZonesDrawing.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PVPARALLELVIEW_PVZONESDRAWING_H
#define PVPARALLELVIEW_PVZONESDRAWING_H

#include <pvkernel/core/general.h>
#include <pvparallelview/common.h>
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

namespace PVParallelView {

template <size_t Bbits>
class PVBCIDrawingBackend;

class PVHSVColor;

template <size_t Bbits = NBITS_INDEX>
class PVZonesDrawing: boost::noncopyable
{
	typedef boost::array<PVBCICode<Bbits>, NBUCKETS> array_codes_t;
	typedef tbb::enumerable_thread_specific<array_codes_t> tls_codes_t;

public:
	typedef PVBCIDrawingBackend<Bbits> bci_backend_t;
	typedef typename bci_backend_t::backend_image_t backend_image_t;
	typedef typename bci_backend_t::backend_image_p_t backend_image_p_t;
	typedef typename bci_backend_t::bci_codes_t bci_codes_t;

public:
	PVZonesDrawing(PVZonesManager& zm, bci_backend_t const& backend, PVHSVColor const& colors):
		_zm(zm),
		_draw_backend(&backend),
		_colors(&colors)
	{
		_computed_codes = PVBCICode<Bbits>::allocate_codes(NBUCKETS);
	}

	~PVZonesDrawing()
	{
		PVBCICode<Bbits>::free_codes(_computed_codes);
	}

/*public:
	template <typename Tree, typename Fbci>
	void draw(QImage& dst_img, Tree const& tree, Fbci const& f_bci);*/

public:
	inline void set_backend(bci_backend_t const& backend) { _draw_backend = &backend; }

public:
	inline PVBCIBackendImage_p<Bbits> create_image(size_t width) const
	{
		assert(_draw_backend);
		PVBCIBackendImage_p<Bbits> ret = _draw_backend->create_image(width);
		ret->set_width(width);
		return ret;
	}

public:
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
			[&](PVZoneTree const& zone_tree, PVHSVColor const* colors, PVBCICode<Bbits>* codes)
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
				// Get thread-local codes buffer
				array_codes_t& arr_codes = this->_tls_computed_codes.local();
				PVBCICode<Bbits>* codes = &arr_codes[0];

				// Get the BCI codes
				PVZoneTree const& zone_tree = _zm.get_zone_tree<PVZoneTree>(zone);
				size_t ncodes = f_bci(zone_tree, _colors, codes);

				// And draw them...
				PVBCIBackendImage<Bbits> &dst_img = *(*(dst_img_begin + zone));
				draw_bci(dst_img, 0, dst_img.width(), codes, ncodes);
			}
		);
	}

	template <class Fbci>
	uint32_t draw_zones(PVBCIBackendImage<Bbits>& dst_img, uint32_t x_start, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			assert(x_start + _zm.get_zone_width(zone) + AxisWidth <= dst_img.width());
			draw_zone<Fbci>(dst_img, x_start, zone, f_bci);
			x_start += _zm.get_zone_width(zone) + AxisWidth;
		}
		return x_start;
	}

	template <class Fbci>
	inline void draw_zone(PVBCIBackendImage<Bbits>& dst_img, uint32_t x_start, PVZoneID zone, Fbci const& f_bci)
	{
		PVZoneTree const &zone_tree = _zm.get_zone_tree<PVZoneTree>(zone);
		draw_bci_lambda<PVZoneTree>(zone_tree, dst_img, x_start, _zm.get_zone_width(zone),
			[&](PVZoneTree const& zone_tree, PVHSVColor const* colors, PVBCICode<Bbits>* codes)
			{
				return (zone_tree.*f_bci)(colors, codes);
			}
	   );
	}

	template <class Fbci>
	inline void draw_zoomed_zone(PVBCIBackendImage<Bbits> &dst_img, uint32_t y_min, int zoom, PVZoneID zone, Fbci const &f_bci, const float zoom_y = 1.0f, const float zoom_x = 1.0f)
	{
		PVZoomedZoneTree const &zoomed_zone_tree = _zm.get_zone_tree<PVZoomedZoneTree>(zone);
		draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
			(zoomed_zone_tree, dst_img, 0, dst_img.width(),
			 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
			     PVParallelView::PVHSVColor const* colors,
			     PVParallelView::PVBCICode<Bbits>* codes)
			 {
				 return (zoomed_zone_tree.*f_bci)(y_min, zoom, colors, codes, zoom_x);
			 }, zoom_y);
	}


	template <class Fbci>
	inline void draw_zoomed_zone(PVBCIBackendImage<Bbits> &dst_img, uint32_t y_min, uint32_t y_max, int zoom, PVZoneID zone, Fbci const &f_bci, const float zoom_y = 1.0f, const float zoom_x = 1.0f)
	{
		PVZoomedZoneTree const &zoomed_zone_tree = _zm.get_zone_tree<PVZoomedZoneTree>(zone);
		draw_bci_lambda<PVParallelView::PVZoomedZoneTree>
			(zoomed_zone_tree, dst_img, 0, dst_img.width(),
			 [&](PVParallelView::PVZoomedZoneTree const &zoomed_zone_tree,
			     PVParallelView::PVHSVColor const* colors,
			     PVParallelView::PVBCICode<Bbits>* codes)
			 {
				 return (zoomed_zone_tree.*f_bci)(y_min, y_max, zoom, colors, codes, zoom_x);
			 }, zoom_y);
	}


	template <class Tree, class Fbci>
	void draw_bci_lambda(Tree const &zone_tree, backend_image_t& dst_img, uint32_t x_start, size_t width, Fbci const& f_bci, const float zoom_y = 1.0f)
	{
		size_t ncodes = f_bci(zone_tree, _colors, _computed_codes);
		draw_bci(dst_img, x_start, width, _computed_codes, ncodes, zoom_y);
	}

	inline uint32_t get_zone_width(PVZoneID z) const
	{
		return _zm.get_zone_width(z);
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
	void draw_bci(backend_image_t& dst_img, uint32_t x_start, size_t width, bci_codes_t* codes, size_t n, const float zoom_y = 1.0f)
	{
		_draw_backend->operator()(dst_img, x_start, width, codes, n, zoom_y);
	}

private:
	PVZonesManager& _zm;
	bci_backend_t const* _draw_backend;
	PVHSVColor const* _colors;
	PVBCICode<Bbits>* _computed_codes;
	tls_codes_t _tls_computed_codes;
};

}

#endif