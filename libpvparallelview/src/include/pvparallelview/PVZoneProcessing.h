/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVPARALLELVIW_PVZONEPROCESSING_H
#define PVPARALLELVIW_PVZONEPROCESSING_H

#include <inendi/PVPlotted.h>

namespace PVParallelView
{

class PVZoneProcessing
{
  public:
	PVZoneProcessing(Inendi::PVPlotted::uint_plotted_table_t const& plotted,
	                 PVRow nrows,
	                 PVCol col_a = 0,
	                 PVCol col_b = 1)
	    : _plotted(plotted)
	    , _nrows(nrows)
	    , _col_a(col_a)
	    , _col_b(col_b)
	    , _nrows_aligned(Inendi::PVPlotted::get_aligned_row_count(nrows))
	{
	}

  public:
	void set_col_a(PVCol c) { _col_a = c; }
	void set_col_b(PVCol c) { _col_b = c; }

  public:
	inline PVRow nrows() const { return _nrows; }
	inline PVCol const& col_a() const { return _col_a; }
	inline PVCol const& col_b() const { return _col_b; }
	inline PVCol& col_a() { return _col_a; }
	inline PVCol& col_b() { return _col_b; }
	inline PVRow nrows_aligned() const { return _nrows_aligned; }

	inline uint32_t get_plotted_value(PVRow r, PVCol c) const { return get_plotted_col(c)[r]; }

	inline uint32_t const* get_plotted_col(PVCol c) const
	{
		return Inendi::PVPlotted::get_plotted_col_addr(&_plotted.at(0), _nrows, c);
	}

	inline uint32_t const* get_plotted_col_a() const { return get_plotted_col(col_a()); }
	inline uint32_t const* get_plotted_col_b() const { return get_plotted_col(col_b()); }

  private:
	Inendi::PVPlotted::uint_plotted_table_t const& _plotted;
	PVRow _nrows;
	PVCol _col_a;
	PVCol _col_b;
	PVRow _nrows_aligned;
};
}

#endif
