/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
#ifdef INSPECTOR_BENCH
static constexpr unsigned int dupl = 200;
#else
static constexpr unsigned int dupl = 1;
#endif

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::View);

	auto view = env.root.get_children<Inendi::PVView>().front();

	/**
	 * Create a second layer without the second element (first element is invalid)
	 */
	view->add_new_layer("layer #2");
	view->set_layer_stack_selected_layer_index(1);

	Inendi::PVSelection sel(view->get_row_count());
	sel.select_all();
	sel.clear_bit_fast(1);
	view->set_selection_view(sel);
	Inendi::PVLayer& current_layer = view->get_current_layer();
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
	         (size_t)(view->get_row_count() - 3));

	view->toggle_listing_unselected_visibility();

	// Selected and unselected lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 2));

	view->toggle_listing_zombie_visibility();

	// Selected, unselected and zombi lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_listing_unselected_visibility();

	// Selected and zombi lines are visible
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 2));

	view->toggle_view_unselected_zombie_visibility();

	//  No modification in listing selection
	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 2));

	return 0;
}
