/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVLINESPROPERTIES_H
#define INENDI_PVLINESPROPERTIES_H

#include <QVector>

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
	explicit PVLinesProperties(size_t size) : _colors(size) {}

	inline PVCore::PVHSVColor const* get_buffer() const { return _colors.data(); }

	void set_line_properties(const PVRow r, PVCore::PVHSVColor c) { _colors[r] = c; }

	/**
	 * Gets the PVHSVColor of a given line
	 *
	 * @param r The index of the line (its row number)
	 */
	inline const PVCore::PVHSVColor get_line_properties(const PVRow r) const { return _colors[r]; }

	void A2B_copy_restricted_by_selection(PVLinesProperties& b, PVSelection const& selection) const;
	void reset_to_default_color();
	void selection_set_color(PVSelection const& selection, const PVCore::PVHSVColor c);

  public:
	void serialize_write(PVCore::PVSerializeObject& so);
	static PVLinesProperties serialize_read(PVCore::PVSerializeObject& so);

  private:
	std::vector<PVCore::PVHSVColor> _colors;
};
}

#endif /* INENDI_PVLINESPROPERTIES_H */
