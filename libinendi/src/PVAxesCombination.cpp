//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <numeric>

#include <inendi/PVAxesCombination.h>

#include <pvkernel/rush/PVFormat.h>

namespace Inendi
{

PVAxesCombination::PVAxesCombination(PVRush::PVFormat const& format)
    : _axes(format.get_axes()), _axes_comb(format.get_axes_comb())
{
}

PVAxesCombination::PVAxesCombination(QList<PVRush::PVAxisFormat> const& axes)
    : _axes(axes), _axes_comb(axes.size())
{
	std::iota(_axes_comb.begin(), _axes_comb.end(), PVCol(0));
}

PVRush::PVAxisFormat const& PVAxesCombination::get_axis(PVCombCol col) const
{
	assert(col != PVCombCol());
	return _axes[_axes_comb[col]];
}

PVRush::PVAxisFormat const& PVAxesCombination::get_axis(PVCol col) const
{
	assert(col != PVCol());
	return _axes[col];
}

PVCol PVAxesCombination::get_nraw_axis(PVCombCol col) const
{
	return col == PVCombCol() ? PVCol() : _axes_comb[col];
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

PVCol PVAxesCombination::get_nraw_axes_count() const
{
	return PVCol(_axes.size());
}

PVCombCol PVAxesCombination::get_axes_count() const
{
	return PVCombCol(_axes_comb.size());
}

PVCombCol PVAxesCombination::get_first_comb_col(PVCol nraw_col) const
{
	auto it = std::find(_axes_comb.begin(), _axes_comb.end(), nraw_col);
	if (it == _axes_comb.end()) {
		return {};
	}

	return PVCombCol(std::distance(_axes_comb.begin(), it));
}

void PVAxesCombination::set_combination(std::vector<PVCol> const& comb)
{
	_axes_comb = comb;
}

void PVAxesCombination::axis_append(PVCol comb_col)
{
	_axes_comb.push_back(comb_col);
}

void PVAxesCombination::delete_axes(PVCol col)
{
	_axes_comb.erase(std::remove(_axes_comb.begin(), _axes_comb.end(), col));
	for (PVCol& c : _axes_comb) {
		if (c > col) {
			c--;
		}
	}
}

void PVAxesCombination::reset_to_default()
{
	_axes_comb.resize(_axes.size());
	std::iota(_axes_comb.begin(), _axes_comb.end(), PVCol(0));
}

bool PVAxesCombination::is_default()
{
	std::vector<PVCol> to_cmp(_axes.size());
	std::iota(to_cmp.begin(), to_cmp.end(), PVCol(0));
	return to_cmp == _axes_comb;
}

void PVAxesCombination::sort_by_name()
{
	std::stable_sort(_axes_comb.begin(), _axes_comb.end(), [this](PVCol c1, PVCol c2) {
		return _axes[c1].get_name() < _axes[c2].get_name();
	});
}

QString PVAxesCombination::to_string() const
{
	QStringList res;
	for (PVCol c : _axes_comb) {
		res << QString::number(c);
	}
	return res.join(",");
}

bool PVAxesCombination::is_last_axis(PVCombCol c) const
{
	return size_t(c + 1) == _axes_comb.size();
}

PVAxesCombination PVAxesCombination::serialize_read(PVCore::PVSerializeObject& so,
                                                    PVRush::PVFormat const& f)
{
	PVAxesCombination comb(f);
	int size = so.attribute_read<int>("size");
	std::vector<PVCol> new_comb(size);
	for (int i = 0; i < size; i++) {
		new_comb[i] = PVCol(so.attribute_read<PVCol::value_type>(QString::number(i)));
	}
	comb.set_combination(new_comb);
	return comb;
}

void PVAxesCombination::serialize_write(PVCore::PVSerializeObject& so) const
{
	int size = _axes_comb.size();
	so.attribute_write("size", size);
	for (size_t i = 0; i < _axes_comb.size(); i++) {
		so.attribute_write(QString::number(i), QVariant(_axes_comb[i].value()));
	}
}
} // namespace Inendi
