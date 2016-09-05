/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <pvkernel/core/inendi_assert.h>
#include <pvkernel/core/PVSerializeArchiveZip.h>
#include <pvkernel/rush/PVNrawCacheManager.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxy.log.format";
static constexpr unsigned int ROW_COUNT = 100000;
#ifdef INSPECTOR_BENCH
static constexpr unsigned int dupl = 200;
#else
static constexpr unsigned int dupl = 1;
#endif

int main()
{
	// Check multiple sources in multipls scene
	pvtest::TestEnv env(csv_file, csv_file_format, dupl, pvtest::ProcessUntil::View);

	auto view = env.root.get_children<Inendi::PVView>().front();

	/**
	 * Create a second layer without the first element
	 */
	view->add_new_layer("layer #2");
	view->set_layer_stack_selected_layer_index(1);

	Inendi::PVSelection sel(view->get_row_count());
	sel.select_all();
	sel.clear_bit_fast(0);
	view->set_selection_view(sel);
	Inendi::PVLayer& current_layer = view->get_current_layer();
	view->commit_selection_to_layer(current_layer);

	/**
	 * Make first layer invisible (so first line is a zombie)
	 */
	view->toggle_layer_stack_layer_n_visible_state(0);
	view->process_layer_stack(sel);

	/**
	 * Unselect the second line.
	 */
	sel.clear_bit_fast(1);
	view->set_selection_view(sel);

	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 2));

	view->toggle_listing_unselected_visibility();

	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_listing_zombie_visibility();

	PV_VALID(view->get_selection_visible_listing().bit_count(), (size_t)(view->get_row_count()));

	view->toggle_listing_unselected_visibility();

	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_view_unselected_zombie_visibility();

	PV_VALID(view->get_selection_visible_listing().bit_count(),
	         (size_t)(view->get_row_count() - 1));

	view->toggle_listing_unselected_visibility();

	PV_VALID(view->get_selection_visible_listing().bit_count(), (size_t)(view->get_row_count()));

	return 0;
}
