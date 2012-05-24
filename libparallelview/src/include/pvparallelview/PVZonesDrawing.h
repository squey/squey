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

class PVBCIDrawingBackend;
class PVHSVColor;

class PVZonesDrawing: public boost::noncopyable
{
	typedef boost::array<PVBCICode, NBUCKETS> array_codes_t;
	typedef tbb::enumerable_thread_specific<array_codes_t> tls_codes_t;

public:
	PVZonesDrawing(PVZonesManager& zm, PVBCIDrawingBackend const& backend, PVHSVColor const& colors);
	~PVZonesDrawing();

/*public:
	template <typename Tree, typename Fbci>
	void draw(QImage& dst_img, Tree const& tree, Fbci const& f_bci);*/

public:
	inline void set_backend(PVBCIDrawingBackend const& backend) { _draw_backend = &backend; }

public:
	inline PVBCIBackendImage_p create_image(size_t width) const { assert(_draw_backend); return _draw_backend->create_image(width); }

public:
	template <class Tree, class Fbci, class BackendImageIterator>
	void draw_zones(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			draw_zone<Tree, Fbci>(dst_img_begin, 0, zone, f_bci);
			dst_img_begin++;
		}
	}
	
	template <class Tree, class Fbci, class BackendImageIterator>
	inline QFuture<void> draw_zones_futur(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		return draw_zones_futur_lambda<Tree>(dst_img_begin, zone_start, nzones,
			[&](Tree const& zone_tree, PVHSVColor const* colors, PVBCICode* codes)
			{
				return (zone_tree.*f_bci)(colors, codes);
			}
	   );
	}

	template <class Tree, class Fbci, class BackendImageIterator>
	QFuture<void> draw_zones_futur_lambda(BackendImageIterator dst_img_begin, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		return QtConcurrent::map(boost::counting_iterator<PVZoneID>(zone_start), boost::counting_iterator<PVZoneID>(nzones),
			[&](PVZoneID zone)
			{
				// Get thread-local codes buffer
				array_codes_t& arr_codes = _tls_computed_codes.local();
				PVBCICode* codes = &arr_codes[0];

				// Get the BCI codes
				Tree const& zone_tree = _zm.get_zone_tree<Tree>(zone);
				size_t ncodes = f_bci(zone_tree, _colors, codes);

				// And draw them...
				draw_bci(*(*(dst_img_begin + zone)), 0, zone, codes, ncodes);
			}
		);
	}


	template <class Tree, class Fbci>
	uint32_t draw_zones(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone_start, PVZoneID nzones, Fbci const& f_bci)
	{
		for (PVZoneID zone = zone_start; zone < nzones; zone++) {
			assert(x_start + _zm.get_zone_width(zone) + AxisWidth <= dst_img.width());
			draw_zone<Tree,Fbci>(dst_img, x_start, zone, f_bci);
			x_start += _zm.get_zone_width(zone) + AxisWidth;
		}
		return x_start;
	}

	template <class Tree, class Fbci>
	void draw_zone_lambda(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, Fbci const& f_bci)
	{
		Tree const& zone_tree = _zm.get_zone_tree<Tree>(zone);
		PVLOG_INFO("draw_zone: tree pointer: %p\n", &zone_tree);
		size_t ncodes = f_bci(zone_tree, _colors, _computed_codes);
		draw_bci(dst_img, x_start, zone, _computed_codes, ncodes);
	}

	template <class Tree, class Fbci>
	inline void draw_zone(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, Fbci const& f_bci)
	{
		draw_zone_lambda<Tree>(dst_img, x_start, zone,
			[&](Tree const& zone_tree, PVHSVColor const* colors, PVBCICode* codes)
			{
				return (zone_tree.*f_bci)(colors, codes);
			}
	   );
	}

	// a = 10; b = 250;
	//draw_zone_lambda<PVParallelView::PVZoomedZoneTree>(img, 0, 4, [&](PVZoomedZoneTree const& zone_tree, PVHSVColor const* colors, PVBCICodes* codes) { zone_tree.create_codes(a, b); });

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
	void draw_bci(PVBCIBackendImage& dst_img, uint32_t x_start, PVZoneID zone, PVBCICode* codes, size_t n);

private:
	PVZonesManager& _zm;
	PVBCIDrawingBackend const* _draw_backend;
	PVHSVColor const* _colors;
	PVBCICode* _computed_codes;
	tls_codes_t _tls_computed_codes;
};

}

#endif
