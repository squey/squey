/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <inendi/PVAxesCombination.h>

namespace Inendi
{

PVAxesCombination::PVAxesCombination(PVRush::PVFormat const& format)
    : _axes(format.get_axes()), _axes_comb(format.get_axes_comb())
{
}

PVAxesCombination::PVAxesCombination(QList<PVRush::PVAxisFormat> const& axes)
    : _axes(axes), _axes_comb(axes.size())
{
	std::iota(_axes_comb.begin(), _axes_comb.end(), 0);
}

PVRush::PVAxisFormat const& PVAxesCombination::get_axis(size_t col) const
{
	return _axes[_axes_comb[col]];
}
PVCol PVAxesCombination::get_nraw_axis(size_t col) const
{
	return _axes_comb[col];
}

std::vector<PVCol> const& PVAxesCombination::get_combination() const
{
	return _axes_comb;
}

QStringList PVAxesCombination::get_nraw_names() const
{
	QStringList l;
	for (PVRush::PVAxisFormat const& fmt : _axes) {
		l << fmt.get_name();
	}
	return l;
}

QStringList PVAxesCombination::get_combined_names() const
{
	QStringList l;
	for (PVCol c : _axes_comb) {
		l << _axes[c].get_name();
	}
	return l;
}

size_t PVAxesCombination::get_axes_count() const
{
	return _axes_comb.size();
}

PVCol PVAxesCombination::get_first_comb_col(PVCol nraw_col) const
{
	auto it = std::find(_axes_comb.begin(), _axes_comb.end(), nraw_col);
	if (it == _axes_comb.end()) {
		return PVCOL_INVALID_VALUE;
	}

	return std::distance(_axes_comb.begin(), it);
}

void PVAxesCombination::set_combination(std::vector<PVCol> const& comb)
{
	_axes_comb = comb;
}

void PVAxesCombination::axis_append(PVCol comb_col)
{
	_axes_comb.push_back(comb_col);
}

void PVAxesCombination::reset_to_default()
{
	_axes_comb.resize(get_axes_count());
	std::iota(_axes_comb.begin(), _axes_comb.end(), 0);
}

bool PVAxesCombination::is_default()
{
	std::vector<PVCol> to_cmp(_axes.size());
	std::iota(to_cmp.begin(), to_cmp.end(), 0);
	return to_cmp == _axes_comb;
}

void PVAxesCombination::sort_by_name()
{
	std::sort(_axes_comb.begin(), _axes_comb.end(),
	          [this](PVCol c1, PVCol c2) { return _axes[c1].get_name() < _axes[c2].get_name(); });
}

QString PVAxesCombination::to_string() const
{
	QStringList res;
	for (PVCol c : _axes_comb) {
		res << QString::number(c);
	}
	return res.join(",");
}

PVAxesCombination PVAxesCombination::serialize_read(PVCore::PVSerializeObject& so,
                                                    PVRush::PVFormat const& f)
{
	PVAxesCombination comb(f);
	int size;
	so.attribute("size", size);
	std::vector<PVCol> new_comb(size);
	for (int i = 0; i < size; i++) {
		so.attribute(QString::number(i), new_comb[i]);
	}
	comb.set_combination(new_comb);
	return comb;
}

void PVAxesCombination::serialize_write(PVCore::PVSerializeObject& so)
{
	int size = _axes_comb.size();
	so.attribute_write("size", size);
	for (size_t i = 0; i < _axes_comb.size(); i++) {
		so.attribute_write(QString::number(i), _axes_comb[i]);
	}
}
}
