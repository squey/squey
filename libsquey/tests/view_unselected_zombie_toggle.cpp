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

#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
#ifdef SQUEY_BENCH
static constexpr unsigned int dupl = 200;
#else
static constexpr unsigned int dupl = 1;
#endif

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::View);

	auto view = env.root.get_children<Squey::PVView>().front();

	/**
	 * Create a second layer without the second element (first element is invalid)
	 */
	view->add_new_layer("layer #2");
	view->set_layer_stack_selected_layer_index(1);

	Squey::PVSelection sel(view->get_row_count());
	sel.select_all();
	sel.clear_bit_fast(1);
	view->set_selection_view(sel);
	Squey::PVLayer& current_layer = view->get_current_layer();
	view->commit_selection_to_layer(current_layer);

	/**
	 * Make first layer invisible (so first line is a zombie)
	 */
	view->toggle_layer_stack_layer_n_visible_state(0);
	view->set_selection_view(sel, true);

	/**
	 * Unselect the third line.
	 */
	sel.clear_bit_fast(2);
	view->set_selection_view(sel);

	// Selected lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 2));

	view->toggle_listing_unselected_visibility();

	// Selected and unselected lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_listing_zombie_visibility();

	// Selected, unselected and zombi lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 0));

	view->toggle_listing_unselected_visibility();

	// Selected and zombi lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_view_unselected_zombie_visibility();

	//  No modification in listing selection
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	return 0;
}
