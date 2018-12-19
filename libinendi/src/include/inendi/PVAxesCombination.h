/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
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
	PVCombCol get_axes_count() const;
	PVCombCol get_first_comb_col(PVCol nraw_col) const;
	QString to_string() const;
	bool is_last_axis(PVCombCol) const;

  public:
	void set_combination(std::vector<PVCol> const& comb);
	void axis_append(PVCol comb_col);
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
