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

#include <squey/PVAnalysisHistory.h>
#include <squey/PVRoot.h>
#include <squey/PVView.h>

#include <pvkernel/core/squey_assert.h>

#include "common.h"

static constexpr const char* csv_file = TEST_FOLDER "/sources/proxy_1bad.log";
static constexpr const char* csv_file_format = TEST_FOLDER "/formats/proxÿ.log.format";

using Scope = Squey::PVAnalysisHistory::Scope;

/** Nothing merges when the window is this short. */
static constexpr std::chrono::milliseconds no_merging{0};

static size_t shown_rows(Squey::PVView const& view)
{
	return view.get_real_output_selection().bit_count();
}

int main()
{
	pvtest::TestEnv env(csv_file, csv_file_format, 1, pvtest::ProcessUntil::View);
	Squey::PVView* view = env.root.get_children<Squey::PVView>().front();
	Squey::PVAnalysisHistory& history = env.root.history();

	size_t changes_seen = 0;
	history._changed.connect([&] { ++changes_seen; });

	const size_t all_rows = shown_rows(*view);
	PV_ASSERT_VALID(all_rows > 0);
	PV_ASSERT_VALID(history.is_empty(), "why", "nothing has been done yet");

	// ------------------------------------------------------------- one step

	{
		Scope step(env.root, "Empty selection", "square");
		view->select_none();
	}

	PV_VALID(history.size(), size_t(2), "why", "the state before the step has to be a step too");
	PV_VALID(history.position(), size_t(1));
	PV_ASSERT_VALID(history.step(1).label() == QString("Empty selection"));
	PV_VALID(history.step(1).selected_row_count(), size_t(0));
	PV_VALID(history.step(0).selected_row_count(), all_rows);
	PV_VALID(changes_seen, size_t(1));
	PV_ASSERT_VALID(history.can_undo() and not history.can_redo());

	// ------------------------------------------------------- undo and redo

	history.undo();
	PV_VALID(history.position(), size_t(0));
	PV_VALID(shown_rows(*view), all_rows, "why", "undoing has to put the rows back");
	PV_ASSERT_VALID(not history.can_undo() and history.can_redo());

	history.redo();
	PV_VALID(history.position(), size_t(1));
	PV_VALID(shown_rows(*view), size_t(0));

	// Landing where one already stands changes nothing and says nothing.
	changes_seen = 0;
	history.go_to(1);
	PV_VALID(changes_seen, size_t(0));

	// ------------------------------------------- a step that changed nothing

	{
		Scope step(env.root, "Opened a dialog and thought better of it", "square");
	}
	PV_VALID(history.size(), size_t(2), "why", "a step that changed nothing must leave no crumb");

	/* Doing again what was already done does leave one, though: a step is
	 * recorded when a value was written to, not when the bits it holds came out
	 * different. Telling those apart would mean comparing every layer of every
	 * view on each action -- a hundred megabytes a step on a large collection --
	 * to spare the breadcrumb an entry for something the user did ask for.
	 */
	{
		Scope step(env.root, "Emptied an already empty selection", "square");
		view->select_none();
	}
	PV_VALID(history.size(), size_t(3));

	// -------------------------------------------------------------- nesting

	{
		Scope outer(env.root, "Whole gesture", "selection-square");
		view->select_all();
		{
			Scope inner(env.root, "Part of it", "square");
			view->select_none();
		}
		view->select_all();
	}
	PV_VALID(history.size(), size_t(4), "why", "nested scopes are one step");
	PV_ASSERT_VALID(history.step(3).label() == QString("Whole gesture"), "why",
	                "the outermost scope names the step");
	PV_VALID(shown_rows(*view), all_rows);

	// -------------------------------------------------------------- merging

	{
		Scope step(env.root, "Dragging", "selection-square", "selection-rectangle");
		view->select_none();
	}
	PV_VALID(history.size(), size_t(5));

	{
		Scope step(env.root, "Dragging some more", "selection-square", "selection-rectangle");
		view->select_all();
	}
	PV_VALID(history.size(), size_t(5), "why", "a drag is one step however often it commits");
	PV_ASSERT_VALID(history.step(4).label() == QString("Dragging"), "why",
	                "the joined step keeps the name of the gesture it started");
	PV_VALID(shown_rows(*view), all_rows, "why", "but it holds the newest state");

	{
		Scope step(env.root, "Something else", "swap", "another-gesture");
		view->select_none();
	}
	PV_VALID(history.size(), size_t(6), "why", "another kind of gesture is another step");

	{
		Scope step(env.root, "Too late to join", "swap", "another-gesture", no_merging);
		view->select_all();
	}
	PV_VALID(history.size(), size_t(7), "why", "a gesture that came too late is its own step");

	// ------------------------------------------- acting after having gone back

	history.go_to(1);
	PV_ASSERT_VALID(history.can_redo());
	{
		Scope step(env.root, "A different turn", "square-check");
		view->select_all();
	}
	PV_VALID(history.size(), size_t(3), "why", "the branch not being followed is let go of");
	PV_VALID(history.position(), size_t(2));
	PV_ASSERT_VALID(not history.can_redo());

	// ------------------------------------------------------------ a barrier

	history.clear();
	PV_ASSERT_VALID(history.is_empty());
	PV_ASSERT_VALID(not history.can_undo() and not history.can_redo());

	// -------------------------------------------- old steps falling off the end

	history.set_max_steps(3);
	for (int i = 0; i < 6; ++i) {
		Scope step(env.root, QString("Step %1").arg(i), "square");
		if (i % 2 == 0) {
			view->select_none();
		} else {
			view->select_all();
		}
	}
	PV_VALID(history.size(), size_t(3), "why", "only the last steps are kept");
	PV_VALID(history.position(), size_t(2), "why", "which is still where the user stands");
	PV_ASSERT_VALID(history.step(2).label() == QString("Step 5"));
	PV_ASSERT_VALID(history.step(0).label() == QString("Step 3"));

	// Whatever fell off, what is left still has to be walkable.
	history.undo();
	PV_VALID(shown_rows(*view), size_t(0));
	history.undo();
	PV_VALID(shown_rows(*view), all_rows);
	PV_ASSERT_VALID(not history.can_undo());

	// --------------------------------- what a view does in reaction to a restore

	/* Putting a state back makes the views react, and a reaction can itself be
	 * an operation: a correlation propagates the restored selection into another
	 * view, under its own scope. That is the step being landed on, not a new one
	 * on top of it, so nothing must be recorded while a restore is under way.
	 */
	history.clear();
	history.set_max_steps(Squey::PVAnalysisHistory::default_max_steps);

	{
		Scope step(env.root, "Something to come back from", "square");
		view->select_none();
	}
	const size_t before_reacting = history.size();

	bool reacting = false;
	auto reaction = view->_selection_view_changed.connect([&] {
		if (reacting) {
			return;
		}
		reacting = true;
		Scope step(env.root, "Propagated by a correlation", "share-all");
		view->set_selection_view(view->get_real_output_selection());
		reacting = false;
	});

	history.undo();
	PV_VALID(history.size(), before_reacting, "why",
	         "what a view does while being restored is not a step of its own");
	PV_VALID(history.position(), size_t(0));

	reaction.disconnect();

	return 0;
}
