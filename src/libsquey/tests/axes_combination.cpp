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

#include <squey/PVAxesCombination.h>

#include <pvkernel/core/squey_assert.h>

#include "common.h"

int main()
{

	init_env();

	PVRush::PVFormat format("my_format", TEST_FOLDER "/picviz/nginx.format");

	Squey::PVAxesCombination axe_comb(format);

	PV_VALID(axe_comb.get_axis(PVCombCol(0)).index.value(), 3);
	PV_VALID(axe_comb.get_nraw_axis(PVCombCol(0)).value(), 3);

	std::vector<size_t> to_compare = {3, 0, 11, 5, 6, 7, 8, 9};
	for (size_t i = 0; i < 8; i++) {
		PV_VALID(axe_comb.get_combination()[i], (PVCol)to_compare[i]);
	}

	QStringList nraw_names = axe_comb.get_nraw_names();
	QStringList ref_nraw_names = {"Source IP", "Axis 2",     "Axis 3",   "Time",          "Axis 5",
	                              "Request",   "url",        "Protocol", "Response code", "Size",
	                              "Axis 9",    "User Agent", "Axis 11"};
	for (int i = 0; i < nraw_names.size(); i++) {
		PV_VALID(nraw_names[i].toStdString(), ref_nraw_names[i].toStdString());
	}

	QStringList comb_names = axe_comb.get_combined_names();
	QStringList ref_comb_names = {"Time", "Source IP", "User Agent",    "Request",
	                              "url",  "Protocol",  "Response code", "Size"};
	for (int i = 0; i < comb_names.size(); i++) {
		PV_VALID(comb_names[i].toStdString(), ref_comb_names[i].toStdString());
	}

	PV_VALID(axe_comb.get_axes_count(), PVCombCol(8));
	PV_ASSERT_VALID(axe_comb.get_first_comb_col(PVCol(5)) == PVCombCol(3));
	PV_VALID(axe_comb.to_string().toStdString(), std::string("3,0,11,5,6,7,8,9"));

	axe_comb.axis_append(PVCol(5));
	PV_VALID(axe_comb.to_string().toStdString(), std::string("3,0,11,5,6,7,8,9,5"));

	std::vector<size_t> to_move = {1, 2, 4};
	axe_comb.move_axes_left_one_position(to_move.begin(), to_move.end());
	PV_VALID(axe_comb.to_string().toStdString(), std::string("0,11,3,6,5,7,8,9,5"));

	std::vector<size_t> to_right_move = {0, 1, 3};
	axe_comb.move_axes_right_one_position(to_right_move.begin(), to_right_move.end());
	PV_VALID(axe_comb.to_string().toStdString(), std::string("3,0,11,5,6,7,8,9,5"));

	axe_comb.remove_axes(to_move.begin(), to_move.end());
	PV_VALID(axe_comb.to_string().toStdString(), std::string("3,5,7,8,9,5"));

	axe_comb.sort_by_name();
	PV_VALID(axe_comb.to_string().toStdString(), std::string("7,5,5,8,9,3"));

	PV_VALID(axe_comb.is_default(), false);

	axe_comb.reset_to_default();
	PV_VALID(axe_comb.to_string().toStdString(), std::string("0,1,2,3,4,5,6,7,8,9,10,11,12"));

	PV_VALID(axe_comb.is_default(), true);

	return 0;
}
