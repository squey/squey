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

#include <squey/PVLayerFilter.h>

#include <pvkernel/core/squey_assert.h>

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>

#include "common.h"

static constexpr const char* csv_file1 = TEST_FOLDER "/sources/proxy_sample1.log";
static constexpr const char* csv_file2 = TEST_FOLDER "/sources/proxy_sample2.log";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/formats/proxy_sample_correlation.format";

#ifdef SQUEY_BENCH
static constexpr unsigned int dupl = 100;
#else
static constexpr unsigned int dupl = 1;
#endif

void run_multiplesearch_filter(Squey::PVView* view1, PVCore::PVOriginalAxisIndexType ait, const PVCore::PVPlainTextType& text_values)
{
	constexpr char plugin_name[] = "search-multiple";
	Squey::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Squey::PVLayerFilter)::get().get_class_by_name(plugin_name);
	Squey::PVLayerFilter::p_type plugin = filter_org->clone<Squey::PVLayerFilter>();
	PVCore::PVArgumentList& args = view1->get_last_args_filter(plugin_name);

	Squey::PVLayer& out = view1->get_post_filter_layer();
	out.reset_to_empty_and_default_color();
	Squey::PVLayer& in = view1->get_layer_stack_output_layer();
	args["axis"].setValue(ait);
	args["exps"].setValue(text_values);

	plugin->set_view(view1);
	plugin->set_output(&out);

	plugin->set_args(args);
	plugin->operator()(in);

	view1->set_selection_view(view1->get_post_filter_layer().get_selection());
}

int main()
{
	pvtest::TestEnv env(csv_file1, csv_file_format, dupl);
	env.add_source(csv_file2, csv_file_format, dupl);

	env.compute_mappings();
	env.compute_plottings();
	env.compute_views();

	auto views = env.root.get_children<Squey::PVView>();
	PV_VALID(views.size(), 2UL);

	/**
	 * Add correlation between source IP columns
	 */
	Squey::PVView* view1 = views.front();
	Squey::PVView* view2 = views.back();

	Squey::PVCorrelation correlation{view1, PVCol(2), view2, PVCol(2)};
	PV_ASSERT_VALID(not env.root.correlations().exists(view1, PVCol(2)));
	PV_ASSERT_VALID(not env.root.correlations().exists(correlation));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) == nullptr);

	env.root.correlations().add(correlation);

	PV_ASSERT_VALID(env.root.correlations().exists(view1, PVCol(2)));
	PV_ASSERT_VALID(
	    env.root.correlations().exists(Squey::PVCorrelation{view1, PVCol(2), view2, PVCol(2)}));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) != nullptr);

	/**
	 * Filter values using multiple-search plugin
	 */
	auto start = std::chrono::system_clock::now();

	run_multiplesearch_filter(
		view1,
		PVCore::PVOriginalAxisIndexType(PVCol(4) /* HTTP status */),
		PVCore::PVPlainTextType("503")
	);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();

#ifndef SQUEY_BENCH
	const Squey::PVSelection& sel1 = view1->get_post_filter_layer().get_selection();
	size_t count1 = sel1.bit_count();
	const Squey::PVSelection& sel2 = view2->get_post_filter_layer().get_selection();
	size_t count2 = sel2.bit_count();
	PV_VALID(count1, 11UL);
	PV_VALID(count2, 84UL);

	std::vector<size_t> v1(
	    {1332, 1485, 1540, 1875, 1877, 1966, 3156, 3159, 3199, 5689, 5762, 5764, 5767, 5791,
	     5843, 6003, 6009, 6177, 6205, 6213, 6221, 6252, 6260, 6293, 6294, 6295, 6297, 6298,
	     6299, 6300, 6301, 6305, 6306, 6307, 6308, 6309, 6311, 6313, 6314, 6315, 6317, 6319,
	     6320, 6321, 6323, 6324, 6325, 6329, 6332, 6333, 6335, 6337, 6340, 6341, 6343, 6344,
	     6346, 6349, 6350, 6354, 6356, 6357, 6358, 6359, 6362, 6365, 6366, 6370, 6374, 6383,
	     6384, 6385, 6410, 6411, 6414, 6420, 9703, 9704, 9747, 9969, 9970, 9974, 9998, 9999});

	std::vector<size_t> v2;
	sel2.visit_selected_lines([&](PVRow const row) { v2.push_back(row); });
	PV_ASSERT_VALID(std::equal(v1.begin(), v1.end(), v2.begin()));

	/**
	 * Add reciprocal correlation
	 */
	Squey::PVCorrelation reciprocal_correlation{view2, correlation.col2, view1, correlation.col1};
	env.root.correlations().add(reciprocal_correlation);
	PV_ASSERT_VALID(env.root.correlations().exists(reciprocal_correlation));
	PV_ASSERT_VALID(env.root.correlations().exists(correlation));

	/**
	 * Remove all correlations (by removing view2 in both ways)
	 */
	env.root.correlations().remove(correlation.view2, true);
	PV_ASSERT_VALID(not env.root.correlations().exists(view1, correlation.col1));
	PV_ASSERT_VALID(not env.root.correlations().exists(view2, reciprocal_correlation.col1));
	PV_ASSERT_VALID(not env.root.correlations().exists(correlation));
	PV_ASSERT_VALID(not env.root.correlations().exists(reciprocal_correlation));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) == nullptr);
	PV_ASSERT_VALID(env.root.correlations().correlation(view2) == nullptr);

	/**
	 * Re-add correlation
	 */
	env.root.correlations().add(correlation);
	PV_ASSERT_VALID(env.root.correlations().exists(view1, PVCol(2)));
	PV_ASSERT_VALID(
	    env.root.correlations().exists(Squey::PVCorrelation{view1, PVCol(2), view2, PVCol(2)}));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) != nullptr);

	/**
	 * Re-remove correlation on view1
	 */
	env.root.correlations().remove(correlation.view1);
	PV_ASSERT_VALID(not env.root.correlations().exists(view1, PVCol(2)));
	PV_ASSERT_VALID(not env.root.correlations().exists(correlation));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) == nullptr);

	/**
	 * Replace correlation
	 */
	Squey::PVCorrelation new_correlation{view1, PVCol(13), view2, PVCol(13)};
	env.root.correlations().add(correlation);
	env.root.correlations().add(new_correlation);
	PV_ASSERT_VALID(not env.root.correlations().exists(correlation));
	PV_ASSERT_VALID(env.root.correlations().exists(new_correlation));

	/**
	 * Check that range correlation is properly working
	 */
	{
	env.root.correlations().remove(correlation.view2, true);
	Squey::PVCorrelation range_correlation{view1, PVCol(5), view2, PVCol(5), Squey::PVCorrelationType::RANGE};
	env.root.correlations().add(range_correlation);
	run_multiplesearch_filter(
		view1,
		PVCore::PVOriginalAxisIndexType(PVCol(5) /* Total bytes */),
		PVCore::PVPlainTextType("0\n999")
	);
	const Squey::PVSelection& sel1 = view1->get_post_filter_layer().get_selection();
	size_t count1 = sel1.bit_count();
	const Squey::PVSelection& sel2 = view2->get_post_filter_layer().get_selection();
	size_t count2 = sel2.bit_count();
	PV_VALID(count1, 138UL);
	PV_VALID(count2, 3990UL);
	}


	/**
	 * Check that removing view1 remove all correlations
	 */
	view1->get_parent().remove_child(*view1);
	PV_ASSERT_VALID(not env.root.correlations().exists(view1, correlation.col1));
	PV_ASSERT_VALID(not env.root.correlations().exists(view2, reciprocal_correlation.col1));
	PV_ASSERT_VALID(not env.root.correlations().exists(correlation));
	PV_ASSERT_VALID(not env.root.correlations().exists(reciprocal_correlation));
	PV_ASSERT_VALID(env.root.correlations().correlation(view1) == nullptr);
	PV_ASSERT_VALID(env.root.correlations().correlation(view2) == nullptr);

#endif

	return 0;
}
