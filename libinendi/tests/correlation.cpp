/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>

#include "common.h"

static constexpr const char* csv_file1 = TEST_FOLDER "/sources/proxy_sample1.log";
static constexpr const char* csv_file2 = TEST_FOLDER "/sources/proxy_sample2.log";
static constexpr const char* csv_file_format =
    TEST_FOLDER "/formats/proxy_sample_correlation.format";

static constexpr unsigned int ROW_COUNT = 10000;

#ifdef INSPECTOR_BENCH
static constexpr unsigned int dupl = 100;
#else
static constexpr unsigned int dupl = 1;
#endif

void run_multiplesearch_filter(Inendi::PVView* view1)
{
	constexpr char plugin_name[] = "search-multiple";
	Inendi::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(plugin_name);
	Inendi::PVLayerFilter::p_type plugin = filter_org->clone<Inendi::PVLayerFilter>();
	PVCore::PVArgumentList& args = view1->get_last_args_filter(plugin_name);

	Inendi::PVLayer& out = view1->get_post_filter_layer();
	out.reset_to_empty_and_default_color(view1->get_row_count());
	Inendi::PVLayer& in = view1->get_layer_stack_output_layer();
	args["axis"].setValue(PVCore::PVOriginalAxisIndexType(4 /* HTTP status */));
	args["exps"].setValue(PVCore::PVPlainTextType("503"));

	plugin->set_view(view1);
	plugin->set_output(&out);

	plugin->set_args(args);
	plugin->operator()(in);

	view1->set_selection_view(view1->get_post_filter_layer().get_selection());

	// explicitely process view to trigger correlation (automatically done by the Hive in Inspector)
	view1->process_from_selection();
}

int main()
{
	pvtest::TestEnv env(csv_file1, csv_file_format, dupl);
	env.add_source(csv_file2, csv_file_format, dupl);

	env.compute_mappings();
	env.compute_plottings();

	std::list<PVCore::PVSharedPtr<Inendi::PVView>> views;
	for (auto scene : env.root->get_children()) {
		for (auto source : scene->get_children()) {
			for (auto mapped : source->get_children()) {
				for (auto plotted : mapped->get_children()) {
					views.insert(views.begin(), plotted->get_children().begin(),
					             plotted->get_children().end());
				}
			}
		}
	}
	PV_VALID(views.size(), 2UL);

	/**
	 * Add correlation between source IP columns
	 */
	Inendi::PVView* view1 = views.front().get();
	Inendi::PVView* view2 = views.back().get();

	Inendi::PVCorrelation correlation{view1, 2, view2, 2};
	PV_ASSERT_VALID(not env.root->correlations().exists(view1, 2));
	PV_ASSERT_VALID(not env.root->correlations().exists(correlation));
	PV_ASSERT_VALID(env.root->correlations().correlation(view1) == nullptr);

	env.root->correlations().add(correlation);

	PV_ASSERT_VALID(env.root->correlations().exists(view1, 2));
	PV_ASSERT_VALID(env.root->correlations().exists(Inendi::PVCorrelation{view1, 2, view2, 2}));
	PV_ASSERT_VALID(env.root->correlations().correlation(view1) != nullptr);

	/**
	 * Filter values using multiple-search plugin
	 */
	auto start = std::chrono::system_clock::now();

	run_multiplesearch_filter(view1);

	auto end = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = end - start;
	return diff.count();

#ifndef INSPECTOR_BENCH
	const Inendi::PVSelection& sel = view2->get_post_filter_layer().get_selection();
	size_t count = sel.get_number_of_selected_lines_in_range(0, sel.count());
	PV_VALID(count, 82UL);

	std::vector<size_t> v1({1332, 1485, 1540, 1875, 1877, 1966, 3156, 3159, 3199, 5689, 5762, 5764,
	                        5767, 5791, 5843, 6003, 6009, 6177, 6205, 6213, 6221, 6252, 6260, 6293,
	                        6294, 6295, 6297, 6298, 6299, 6300, 6301, 6305, 6306, 6307, 6308, 6309,
	                        6311, 6313, 6314, 6315, 6317, 6319, 6320, 6321, 6323, 6324, 6325, 6329,
	                        6332, 6333, 6335, 6337, 6340, 6341, 6343, 6344, 6346, 6349, 6350, 6354,
	                        6356, 6357, 6358, 6359, 6362, 6365, 6366, 6370, 6374, 6383, 6384, 6385,
	                        6410, 6411, 6414, 6420, 9703, 9704, 9747, 9969, 9970, 9974});

	std::vector<size_t> v2;
	sel.visit_selected_lines([&](PVRow const row) { v2.push_back(row); });
	PV_ASSERT_VALID(std::equal(v1.begin(), v1.end(), v2.begin()));

	/**
	 * Remove correlation
	 */
	env.root->correlations().remove(correlation.view1);
	PV_ASSERT_VALID(not env.root->correlations().exists(view1, 2));
	PV_ASSERT_VALID(not env.root->correlations().exists(correlation));
	PV_ASSERT_VALID(env.root->correlations().correlation(view1) == nullptr);

	/**
	 * Replace correlation
	 */
	Inendi::PVCorrelation new_correlation{view1, 13, view2, 13};
	env.root->correlations().add(correlation);
	env.root->correlations().add(new_correlation);
	PV_ASSERT_VALID(not env.root->correlations().exists(correlation));
	PV_ASSERT_VALID(env.root->correlations().exists(new_correlation));

#endif

	return 0;
}
