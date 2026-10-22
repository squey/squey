//
// MIT License
//
// © Squey, 2026
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The rows a sample of a series view stands for, as the values under the cursor
// read them: those of the smallest and of the largest value, among the selected
// rows, and in its group once the timeseries are split.

#include <squey/PVRangeSubSampler.h>
#include <pvkernel/core/squey_assert.h>

#include <QTemporaryDir>

#include <fstream>
#include <string>
#include <vector>

#include "common.h"

namespace
{

constexpr const char* FORMAT = R"(<?xml version='1.0' encoding='UTF-8'?>
<!DOCTYPE PVParamXml>
<param version="11" first_line="0">
 <splitter type="csv" sep="," quote="&quot;">
  <field>
   <axis name="time" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="value" type="number_uint32">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
  <field>
   <axis name="group" type="string">
    <mapping mode="default"/>
    <scaling mode="default"/>
   </axis>
  </field>
 </splitter>
</param>
)";

const PVCol TIME(0);
const PVCol VALUE(1);
const PVCol GROUP(2);

constexpr PVRow ROW_COUNT = 8;

/**
 * The rows of the smallest and of the largest value, as "smallest-largest", or "none".
 */
std::string
extrema(const Squey::PVRangeSubSampler& sampler, size_t sample, size_t timeserie, size_t radius = 0)
{
	const auto rows = sampler.sample_extrema(sample, timeserie, radius);
	return rows ? std::to_string(rows->first) + "-" + std::to_string(rows->second) : "none";
}

} // namespace

int main()
{
	QTemporaryDir dir;
	PV_ASSERT_VALID(dir.isValid());
	const std::string csv = dir.filePath("samples.csv").toStdString();
	const std::string format = dir.filePath("samples.csv.format").toStdString();
	{
		// One row per instant, in two groups taking turns. The instants are far enough apart
		// for the samples below to part them however their bounds are rounded.
		const int values[ROW_COUNT] = {5, 1, 7, 3, 2, 8, 6, 4};
		std::ofstream out(csv);
		for (PVRow row = 0; row < ROW_COUNT; row++) {
			out << 100 * row << ',' << values[row] << ',' << (row % 2 == 0 ? 'a' : 'b') << '\n';
		}
		std::ofstream(format) << FORMAT;
	}

	pvtest::TestEnv env(csv, format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	const PVRush::PVNraw& nraw = view->get_rushnraw_parent();

	std::vector<pvcop::core::array<uint32_t>> timeseries;
	for (const auto& scaled : env.root.get_children<Squey::PVScaled>().front()->get_scaleds()) {
		timeseries.emplace_back(scaled.to_core_array<uint32_t>());
	}
	auto& selection = const_cast<Squey::PVSelection&>(view->get_real_output_selection());
	selection.select_all();
	Squey::PVRangeSubSampler sampler(nraw.column(TIME), timeseries, nraw, selection);
	sampler.set_selected_timeseries({size_t(VALUE)});

	// Two samples of four rows.
	sampler.set_sampling_count(2);
	sampler.resubsample();
	PV_VALID(sampler.histogram()[0], size_t(4));
	PV_VALID(extrema(sampler, 0, VALUE), std::string("1-2")); // 5, 1, 7, 3
	PV_VALID(extrema(sampler, 1, VALUE), std::string("4-5")); // 2, 8, 6, 4
	PV_VALID(extrema(sampler, 2, VALUE), std::string("none"));

	// The rows out of the selection no longer count.
	selection.set_line(2, false);
	selection.set_line(5, false);
	PV_VALID(extrema(sampler, 0, VALUE), std::string("1-0")); // 5, 1, 3
	PV_VALID(extrema(sampler, 1, VALUE), std::string("4-6")); // 2, 6, 4
	selection.select_all();

	// As many samples as rows: one row each.
	sampler.set_sampling_count(ROW_COUNT);
	sampler.resubsample();
	for (PVRow row = 0; row < ROW_COUNT; row++) {
		PV_VALID(sampler.histogram()[row], size_t(1), "sample", row);
		PV_VALID(extrema(sampler, row, VALUE), std::to_string(row) + "-" + std::to_string(row),
		         "sample", row);
	}

	// More samples than rows: some hold none.
	sampler.set_sampling_count(2 * ROW_COUNT);
	sampler.resubsample();
	size_t empty_samples = 0;
	for (size_t sample = 0; sample < sampler.samples_count(); sample++) {
		if (sampler.histogram()[sample] == 0) {
			PV_VALID(extrema(sampler, sample, VALUE), std::string("none"), "sample", sample);
			empty_samples++;
		}
	}
	PV_ASSERT_VALID(empty_samples > 0);

	// Around a sample that holds none, the closest one that holds some, within the radius.
	PV_VALID(sampler.histogram()[1], size_t(0));
	PV_VALID(extrema(sampler, 1, VALUE, 1), std::string("0-0")); // the earlier one first
	PV_VALID(sampler.histogram()[7] + sampler.histogram()[8], size_t(0));
	PV_VALID(extrema(sampler, 8, VALUE, 1), std::string("4-4"));
	PV_VALID(extrema(sampler, 8, VALUE, 0), std::string("none"));

	// Split, a timeserie is a column in a group, and counts the rows of its group only.
	sampler.set_split_column(&nraw.column(GROUP));
	sampler.set_sampling_count(2);
	sampler.resubsample();
	PV_VALID(sampler.group_count(), size_t(2));
	PV_VALID(sampler.group_name(0), std::string("a"));
	const size_t value_a = sampler.group_count() * size_t(VALUE.value());
	const size_t value_b = value_a + 1;
	PV_VALID(extrema(sampler, 0, value_a), std::string("0-2")); // 5, 7
	PV_VALID(extrema(sampler, 1, value_a), std::string("4-6")); // 2, 6
	PV_VALID(extrema(sampler, 0, value_b), std::string("1-3")); // 1, 3
	PV_VALID(extrema(sampler, 1, value_b), std::string("7-5")); // 8, 4

	return 0;
}
