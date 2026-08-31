/* * MIT License
 *
 * © Squey, 2026
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <squey/PVView.h>
#include <squey/PVViewState.h>

#include <pvkernel/core/squey_assert.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxÿ.log.format";

/**
 * The rows a view actually shows, which is what has to come back when a state
 * does: it is computed from the three captured values rather than remembered
 * with them.
 */
static size_t shown_rows(Squey::PVView const& view)
{
	return view.get_real_output_selection().bit_count();
}

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();

	const PVRow row_count = view->get_row_count();
	PV_ASSERT_VALID(row_count > 4);

	// ---------------------------------------------------------------- capture

	const Squey::PVViewState initial = view->capture_state();
	PV_ASSERT_VALID(initial.is_valid());
	PV_ASSERT_VALID(view->capture_state() == initial, "why",
	                "capturing twice without touching anything must give the same state");

	const size_t initially_shown = shown_rows(*view);

	// ------------------------------------------------- a step over the selection

	Squey::PVSelection sel(row_count);
	sel.select_all();
	sel.clear_bit_fast(1);
	sel.clear_bit_fast(2);
	view->set_selection_view(sel);

	const Squey::PVViewState after_selection = view->capture_state();
	PV_ASSERT_VALID(not after_selection.same_selection_as(initial));
	PV_ASSERT_VALID(after_selection.same_layer_stack_as(initial), "why",
	                "changing the selection must leave the layers shared");
	PV_ASSERT_VALID(after_selection.same_axes_combination_as(initial), "why",
	                "changing the selection must leave the axes shared");

	const size_t shown_after_selection = shown_rows(*view);
	PV_ASSERT_VALID(shown_after_selection < initially_shown);

	// ------------------------------------------------------- a step over the layers

	view->add_new_layer("second layer");

	const Squey::PVViewState after_layer = view->capture_state();
	PV_ASSERT_VALID(not after_layer.same_layer_stack_as(after_selection));
	PV_ASSERT_VALID(after_layer.same_selection_as(after_selection), "why",
	                "adding a layer must leave the selection shared");
	PV_VALID(view->get_layer_stack().get_layer_count(), 2);

	// ------------------------------------------- one layer changed, one layer copied

	/* What the stack holding its layers one by one buys: a step that changes a
	 * layer duplicates that layer, not the others. Told by address, since a
	 * layer that was not touched is the very same object a captured state still
	 * points at.
	 */
	{
		const Squey::PVLayer* untouched = &view->get_layer_stack().get_layer_n(0);
		const Squey::PVLayer* renamed = &view->get_layer_stack().get_layer_n(1);
		const Squey::PVViewState before = view->capture_state();

		view->set_layer_stack_layer_n_name(1, "renamed");

		PV_ASSERT_VALID(&view->get_layer_stack().get_layer_n(0) == untouched, "why",
		                "a layer nobody wrote to must not be copied");
		PV_ASSERT_VALID(&view->get_layer_stack().get_layer_n(1) != renamed, "why",
		                "the layer that was written to has to leave the captured one alone");
		PV_ASSERT_VALID(not view->capture_state().same_layer_stack_as(before));

		// And what the state captured still reads as it did.
		view->restore_state(before);
		PV_ASSERT_VALID(view->get_layer_stack().get_layer_n(1).get_name() != QString("renamed"));
	}

	// -------------------------------------------------------------- going back

	view->restore_state(after_selection);
	PV_VALID(view->get_layer_stack().get_layer_count(), 1);
	PV_VALID(shown_rows(*view), shown_after_selection, "why",
	         "what the view shows has to be recomputed, not just the values put back");
	PV_ASSERT_VALID(view->capture_state() == after_selection);

	view->restore_state(initial);
	PV_VALID(shown_rows(*view), initially_shown);
	PV_ASSERT_VALID(view->capture_state() == initial);

	// Landing on the same step twice must be as good as landing on it once.
	view->restore_state(initial);
	PV_VALID(shown_rows(*view), initially_shown);

	// ------------------------------------------------------------ going forward

	view->restore_state(after_layer);
	PV_VALID(view->get_layer_stack().get_layer_count(), 2);
	PV_VALID(shown_rows(*view), shown_after_selection);

	// ------------------------------------------------- moving on from a restored state

	// The states captured earlier must survive what happens after the history
	// has been rewound: this is the redo branch being dropped, not the steps
	// themselves being rewritten.
	view->restore_state(initial);
	view->select_none();
	PV_VALID(shown_rows(*view), size_t(0));

	view->restore_state(after_layer);
	PV_VALID(view->get_layer_stack().get_layer_count(), 2);
	PV_VALID(shown_rows(*view), shown_after_selection);

	view->restore_state(initial);
	PV_VALID(shown_rows(*view), initially_shown, "why",
	         "the first state must still hold what it captured");

	// ---------------------------------------------- a step over the axes combination

	const Squey::PVViewState before_axes = view->capture_state();
	const PVCombCol axes_count = view->get_axes_combination().get_axes_count();
	PV_ASSERT_VALID(axes_count > PVCombCol(1));

	std::vector<PVCol> comb = view->get_axes_combination().get_combination();
	comb.pop_back();
	view->set_axes_combination(comb);

	const Squey::PVViewState after_axes = view->capture_state();
	PV_ASSERT_VALID(not after_axes.same_axes_combination_as(before_axes));
	PV_ASSERT_VALID(after_axes.same_selection_as(before_axes), "why",
	                "changing the axes must leave the selection shared");
	PV_ASSERT_VALID(after_axes.same_layer_stack_as(before_axes));
	PV_VALID(view->get_axes_combination().get_axes_count(), axes_count - PVCombCol(1));

	view->restore_state(before_axes);
	PV_VALID(view->get_axes_combination().get_axes_count(), axes_count);

	return 0;
}
