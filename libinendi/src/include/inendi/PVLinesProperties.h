/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLINESPROPERTIES_H
#define INENDI_PVLINESPROPERTIES_H

#include <QVector>

#include <inendi/general.h>

#include <pvkernel/core/PVHSVColor.h>
#include <pvkernel/core/PVSerializeArchive.h>
#include <inendi/PVSelection.h>

namespace Inendi
{

/**
 * \class PVLinesProperties
 */
class PVLinesProperties
{
	friend class PVCore::PVSerializeObject;

  public:
	typedef std::allocator<PVCore::PVHSVColor> color_allocator_type;
	typedef PVCore::PVHSVColor::h_type h_type;

  public:
	void ensure_allocated(size_t row_count);
	void ensure_initialized(size_t row_count);

	inline PVCore::PVHSVColor const* get_buffer() const { return _colors.data(); }

	void set_line_properties(const PVRow r, PVCore::PVHSVColor c) { _colors[r] = c; }

	/**
	 * Gets the PVHSVColor of a given line
	 *
	 * @param r The index of the line (its row number)
	 */
	inline const PVCore::PVHSVColor get_line_properties(const PVRow r) const { return _colors[r]; }

	void set_row_count(int row_count)
	{
		assert(row_count < INENDI_LINES_MAX);
		_colors.resize(row_count);
	}

	void A2B_copy_restricted_by_selection_and_nelts(PVLinesProperties& b,
	                                                PVSelection const& selection,
	                                                PVRow nelts) const;
	void reset_to_default_color(PVRow row_count);
	void selection_set_color(PVSelection const& selection,
	                         const PVRow nelts,
	                         const PVCore::PVHSVColor c);

  protected:
	void serialize(PVCore::PVSerializeObject& so, PVCore::PVSerializeArchive::version_t v);

  private:
	std::vector<PVCore::PVHSVColor> _colors;
};
}

#endif /* INENDI_PVLINESPROPERTIES_H */
