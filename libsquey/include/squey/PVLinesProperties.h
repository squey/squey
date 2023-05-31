/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SQUEY_PVLINESPROPERTIES_H
#define SQUEY_PVLINESPROPERTIES_H

#include <pvkernel/core/PVHSVColor.h> // for PVHSVColor, etc

#include <pvbase/types.h> // for PVRow

#include <cstddef> // for size_t
#include <vector>  // for vector, allocator

namespace Squey
{
class PVSelection;
} // namespace Squey
namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore

namespace Squey
{

/**
 * \class PVLinesProperties
 */
class PVLinesProperties
{
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
	void serialize_write(PVCore::PVSerializeObject& so) const;
	static PVLinesProperties serialize_read(PVCore::PVSerializeObject& so);

  private:
	std::vector<PVCore::PVHSVColor> _colors;
};
} // namespace Squey

#endif /* SQUEY_PVLINESPROPERTIES_H */
