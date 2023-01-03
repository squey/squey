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

#ifndef INENDI_PVAXESCOMBINATION_H
#define INENDI_PVAXESCOMBINATION_H

#include <pvbase/types.h> // for PVCol

#include <QString>     // for QString
#include <QStringList> // for QStringList

#include <vector>
#include <cstddef> // for size_t

namespace PVCore
{
class PVSerializeObject;
} // namespace PVCore
namespace PVRush
{
class PVAxisFormat;
} // namespace PVRush
namespace PVRush
{
class PVFormat;
} // namespace PVRush

template <typename T>
class QList;

namespace Inendi
{

class PVAxesCombination
{
  public:
	explicit PVAxesCombination(PVRush::PVFormat const& format);
	explicit PVAxesCombination(QList<PVRush::PVAxisFormat> const& axes);

  public:
	PVRush::PVAxisFormat const& get_axis(PVCombCol col) const;
	PVRush::PVAxisFormat const& get_axis(PVCol col) const;

	PVCol get_nraw_axis(PVCombCol col) const;
	std::vector<PVCol> const& get_combination() const;
	QStringList get_nraw_names() const;
	QStringList get_combined_names() const;
	PVCol get_nraw_axes_count() const;
	PVCombCol get_axes_count() const;
	PVCombCol get_first_comb_col(PVCol nraw_col) const;
	QString to_string() const;
	bool is_last_axis(PVCombCol) const;

  public:
	void set_combination(std::vector<PVCol> const& comb);
	void axis_append(PVCol comb_col);
	void delete_axes(PVCol col);
	void reset_to_default();
	bool is_default();
	void sort_by_name();

	template <class It>
	void move_axes_left_one_position(It const& begin, It const& end)
	{
		for (auto it = begin; it != end; ++it) {
			std::swap(_axes_comb[*it], _axes_comb[*it - 1]);
		}
	}

	template <class It>
	void move_axes_right_one_position(It const& begin, It const& end)
	{
		for (auto it = end; it != begin; --it) {
			std::swap(_axes_comb[*(it - 1)], _axes_comb[*(it - 1) + 1]);
		}
	}

	template <class It>
	void remove_axes(It const& begin, It const& end)
	{
		for (auto it = end - 1; it != begin - 1; --it) {
			_axes_comb.erase(_axes_comb.begin() + *it);
		}
	}

  public:
	static PVAxesCombination serialize_read(PVCore::PVSerializeObject&, PVRush::PVFormat const&);
	void serialize_write(PVCore::PVSerializeObject&) const;

  private:
	QList<PVRush::PVAxisFormat> const& _axes; //!< View from the PVFormat
	std::vector<PVCol> _axes_comb;            // FIXME : don't hardcode and duplicate this type !!
};
} // namespace Inendi

#endif /* INENDI_PVAXESCOMBINATION_H */
