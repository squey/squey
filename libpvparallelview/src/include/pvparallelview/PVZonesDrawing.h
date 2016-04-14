/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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

#include <tbb/enumerable_thread_specific.h>

#include <cassert>
#include <functional>
#include <array>

namespace PVCore {
class PVHSVColor;
}

namespace PVParallelView {

class PVBCIDrawingBackend;

template <size_t Bbits>
class PVZonesDrawing
{
	typedef std::array<PVBCICodeBase, NBUCKETS> array_codes_t;
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

	PVZonesDrawing(PVZonesDrawing const&) = delete;
	PVZonesDrawing(PVZonesDrawing &&) = delete;
	PVZonesDrawing& operator=(PVZonesDrawing const&) = delete;
	PVZonesDrawing& operator=(PVZonesDrawing &&) = delete;
	~PVZonesDrawing() { }

public:
	template <class Tree, class Fbci>
	void draw_bci_lambda(Tree const &zone_tree, backend_image_t& dst_img, uint32_t x_start, size_t width, Fbci const& f_bci, const float zoom_y, bool reverse = false)
	{
		// Get a free BCI buffers
		PVBCICode<Bbits>* bci_buf = &_computed_codes.get_available_buffer()->as<Bbits>();
		size_t ncodes = f_bci(zone_tree, _colors, bci_buf);
		draw_bci(dst_img, x_start, width, bci_buf, ncodes, zoom_y, reverse,
				[=]
				{
					_computed_codes.return_buffer(reinterpret_cast<PVBCICodeBase*>(bci_buf));
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

private:
	// BCI buffers are shared among all instances of PVZonesDrawing
	static PVBCIBuffers<BCI_BUFFERS_COUNT> _computed_codes;
};

}

#endif
