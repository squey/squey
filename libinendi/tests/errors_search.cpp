/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2015-2015
 */
#include "common.h"

#include <inendi/PVLayerFilter.h>
#include <inendi/PVLayer.h>

#include <pvkernel/core/PVEnumType.h>
#include <pvkernel/core/PVOriginalAxisIndexType.h>
#include <pvkernel/core/PVPlainTextType.h>

#include <pvkernel/core/inendi_assert.h>

constexpr char FILENAME[] = TEST_FOLDER "/picviz/errors_search.csv";

#ifdef INSPECTOR_BENCH
constexpr size_t DUPL = 1000;
#else
constexpr size_t DUPL = 1;
#endif

constexpr char FORMAT[] = TEST_FOLDER "/picviz/errors_search.csv.format";

using options_t = std::array<int, 3>;
using testcase_t = std::pair<options_t, size_t>;

void set_args(PVCore::PVArgumentList& args, const options_t& values)
{
	static constexpr const char* params[] = {"axis", "type", "include"};

	for (size_t i = 0; i < values.size(); i++) {
		if (i == 0) {
			PVCore::PVOriginalAxisIndexType e(values[i]);
			args[params[i]].setValue(e);
		} else {
			PVCore::PVEnumType e = args[params[i]].value<PVCore::PVEnumType>();
			e.set_sel(values[i]);
			args[params[i]].setValue(e);
		}
	}
}

void run_tests(Inendi::PVLayerFilter::p_type& plugin,
               PVCore::PVArgumentList& args,
               const Inendi::PVLayer& in,
               Inendi::PVLayer& out)
{
	std::vector<testcase_t> tests{{{{0 /*axis*/, 0 /*type*/, 0 /*include*/}}, 3},
	                              {{{0 /*axis*/, 0 /*type*/, 1 /*include*/}}, 7},
	                              {{{0 /*axis*/, 1 /*type*/, 0 /*include*/}}, 4},
	                              {{{0 /*axis*/, 1 /*type*/, 1 /*include*/}}, 4},
	                              {{{0 /*axis*/, 2 /*type*/, 0 /*include*/}}, 7},
	                              {{{0 /*axis*/, 2 /*type*/, 1 /*include*/}}, 4},
	                              {{{1 /*axis*/, 0 /*type*/, 0 /*include*/}}, 0},
	                              {{{1 /*axis*/, 0 /*type*/, 1 /*include*/}}, 7},
	                              {{{1 /*axis*/, 1 /*type*/, 0 /*include*/}}, 0},
	                              {{{1 /*axis*/, 1 /*type*/, 1 /*include*/}}, 7},
	                              {{{1 /*axis*/, 2 /*type*/, 0 /*include*/}}, 0},
	                              {{{1 /*axis*/, 2 /*type*/, 1 /*include*/}}, 7}};

	for (const testcase_t& test : tests) {
		set_args(args, test.first);
		plugin->set_args(args);
		if (test.first[2]) { // exclude
			Inendi::PVLayer in_l = in;
			Inendi::PVSelection& s = in_l.get_selection();
			s.select_odd();
			plugin->operator()(in_l);
		} else {
			plugin->operator()(in);
		}
		PV_VALID(out.get_selection().bit_count(), test.second * DUPL);
	}
}

int main()
{
	// Init nraw
	pvtest::TestEnv env(FILENAME, FORMAT, DUPL, pvtest::ProcessUntil::View);
	Inendi::PVView* view = env.root.current_view();

	// Get plugin reference
	constexpr char plugin_name[] = "errors-search";
	Inendi::PVLayerFilter::p_type filter_org =
	    LIB_CLASS(Inendi::PVLayerFilter)::get().get_class_by_name(plugin_name);
	Inendi::PVLayerFilter::p_type fclone = filter_org->clone<Inendi::PVLayerFilter>();
	PVCore::PVArgumentList& args = view->get_last_args_filter(plugin_name);

	Inendi::PVLayer out("Out", view->get_row_count());
	out.reset_to_empty_and_default_color();
	const Inendi::PVLayer& in = view->get_layer_stack_output_layer();

	fclone->set_view(view);
	fclone->set_output(&out);

	auto start = std::chrono::system_clock::now();
	run_tests(fclone, args, in, out);
	auto end = std::chrono::system_clock::now();

	std::chrono::duration<double> diff = end - start;
	std::cout << diff.count();
}
