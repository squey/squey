/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2016
 */

#ifndef PVPARALLELVIW_PVZONEPROCESSING_H
#define PVPARALLELVIW_PVZONEPROCESSING_H

namespace PVParallelView
{

/**
 * Data with plotted information about a given zone.
 */
struct PVZoneProcessing {
	const size_t size;
	uint32_t const* const plotted_a;
	uint32_t const* const plotted_b;
};
} // namespace PVParallelView

#endif
