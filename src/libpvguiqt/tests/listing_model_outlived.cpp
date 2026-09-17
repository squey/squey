//
// MIT License
//
// © Squey, 2026
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

// A listing closed while the view it listed goes on.
//
// The model connects to the view it lists, which outlives it: a listing opened in a
// dock is destroyed as the dock is closed. Two of those connections were lambdas
// capturing the model, which sigc::trackable does not disconnect. A selection change
// then posted an update to the model that was gone -- to the application, which
// outlives everything, rather than to the model -- and deleting the view wrote into
// the model's memory. A model gone has to leave nothing connected behind it.

#include <pvkernel/core/squey_assert.h>

#include <squey/PVRoot.h>
#include <squey/PVSelection.h>
#include <squey/PVSource.h>
#include <squey/PVView.h>

#include <pvguiqt/PVListingModel.h>

#include <QApplication>
#include <QTemporaryDir>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "common.h"
#include "test-env.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="rising" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="label" type="string">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

constexpr PVRow ROWS = 1000;

} // namespace

int main(int argc, char** argv)
{
	init_env();

	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid(), "a temporary directory", "could not be made");
	const QString csv = dir.filePath("rows.csv");
	const QString format = dir.filePath("rows.csv.format");
	{
		std::ofstream out(csv.toStdString());
		for (PVRow i = 0; i < ROWS; ++i) {
			out << i << ",row" << i % 13 << "\n";
		}
		std::ofstream(format.toStdString()) << FORMAT;
	}

	auto root = std::make_unique<Squey::PVRoot>();
	Squey::PVSource& src = get_src_from_file(*root, csv, format);
	src.emplace_add_child()   // Mapped
	    .emplace_add_child()  // Scaled
	    .emplace_add_child(); // View
	Squey::PVView& view = *src.current_view();

	QApplication app(argc, argv);

	const auto selection_slots = view._update_output_selection.size();
	const auto deletion_slots = view._about_to_be_delete.size();

	auto* model = new PVGuiQt::PVListingModel(view);
	PV_ASSERT_VALID(view._update_output_selection.size() > selection_slots,
	                "the model", "does not follow the selection");
	PV_ASSERT_VALID(view._about_to_be_delete.size() > deletion_slots, "the model",
	                "does not follow the view's deletion");

	// A selection changes, and the listing is closed before the update this posted
	// has run.
	Squey::PVSelection half(view.get_row_count());
	half.select_none();
	for (PVRow r = 0; r < view.get_row_count(); r += 2) {
		half.set_bit_fast(r);
	}
	view.set_selection_view(half);
	delete model;

	std::cout << "slots left on the view: selection " << view._update_output_selection.size()
	          << " (was " << selection_slots << "), deletion " << view._about_to_be_delete.size()
	          << " (was " << deletion_slots << ")" << std::endl;
	PV_ASSERT_VALID(view._update_output_selection.size() == selection_slots,
	                "selection handlers left by the model",
	                view._update_output_selection.size() - selection_slots);
	PV_ASSERT_VALID(view._about_to_be_delete.size() == deletion_slots,
	                "deletion handlers left by the model",
	                view._about_to_be_delete.size() - deletion_slots);

	// What was posted runs with the model gone, then another selection change, and
	// the view itself goes.
	QApplication::processEvents();
	view.select_all();
	QApplication::processEvents();
	root.reset();
	QApplication::processEvents();

	return 0;
}
