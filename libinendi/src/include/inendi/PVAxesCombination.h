/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef INENDI_PVAXESCOMBINATION_H
#define INENDI_PVAXESCOMBINATION_H

#include <QStringList>

#include <vector>

#include <pvkernel/core/PVSerializeArchive.h>
#include <pvkernel/rush/PVFormat.h>

#include <inendi/PVCombCol.h>

namespace Inendi
{

class PVAxesCombination
{
  public:
	PVAxesCombination(PVRush::PVFormat const& format);
	PVAxesCombination(QList<PVRush::PVAxisFormat> const& axes);

  public:
	PVRush::PVAxisFormat const& get_axis(PVCombCol col) const;
	PVCol get_nraw_axis(PVCombCol col) const;
	std::vector<PVCol> const& get_combination() const;
	QStringList get_nraw_names() const;
	QStringList get_combined_names() const;
	size_t get_axes_count() const;
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
	void serialize_write(PVCore::PVSerializeObject&);

  private:
	QList<PVRush::PVAxisFormat> const& _axes; //!< View from the PVFormat
	std::vector<PVCol> _axes_comb;
};
}

#endif /* INENDI_PVAXESCOMBINATION_H */
