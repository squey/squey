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

#ifndef SQUEY_PVVIEWSTATE_H
#define SQUEY_PVVIEWSTATE_H

#include <squey/PVAxesCombination.h>
#include <squey/PVLayerStack.h>
#include <squey/PVSelection.h>

#include <pvkernel/core/PVCowValue.h>

namespace Squey
{

class PVView;

/**
 * \class PVViewState
 *
 * What of a view an undo step remembers: the selection, the layer stack and the
 * axes combination. Everything else a view holds is either derived from those
 * three -- the output layers are recomputed from them -- or is presentation the
 * history has no business rewinding, such as which lines the listing shows.
 *
 * Capturing one is a handful of refcount increments whatever the data weighs,
 * and two states that were captured either side of an unchanged value share it
 * rather than each holding a copy. See PVCore::PVCowValue.
 *
 * States compare by identity, not by contents: two states are the same when
 * they hold the very same values. Selections that happen to have the same bits
 * set are therefore different states, which is what the history wants -- it is
 * a record of what the user did, not of what the data happened to look like.
 */
class PVViewState
{
	friend class PVView;

  public:
	PVViewState() = default;

  public:
	/**
	 * Whether this state was captured from a view, as opposed to default-built.
	 */
	bool is_valid() const { return _selection != nullptr; }

	bool operator==(PVViewState const& o) const = default;

	/**
	 * Whether both states leave the analysis where the other does.
	 *
	 * The selection is compared by the rows it holds rather than by identity,
	 * because acts that repeat it are ordinary -- pressing "select all" twice
	 * -- and the second one leaves nothing to come back to. The layers and the
	 * axes are still compared by identity: telling two layer stacks apart by
	 * their contents would mean walking a hundred megabytes on a large
	 * collection, to spare the breadcrumb an entry nobody asked twice for.
	 */
	bool holds_same_as(PVViewState const& o) const
	{
		if (_layer_stack != o._layer_stack || _axes_combination != o._axes_combination) {
			return false;
		}
		if (_selection == o._selection) {
			return true;
		}
		return _selection && o._selection && *_selection == *o._selection;
	}

  public:
	/* What is shared with another state, so that restoring notifies only the
	 * views that have something to hear about.
	 */
	bool same_selection_as(PVViewState const& o) const { return _selection == o._selection; }
	bool same_layer_stack_as(PVViewState const& o) const { return _layer_stack == o._layer_stack; }
	bool same_axes_combination_as(PVViewState const& o) const
	{
		return _axes_combination == o._axes_combination;
	}

  private:
	PVCore::PVCowValue<PVSelection>::snapshot_type _selection;
	PVCore::PVCowValue<PVLayerStack>::snapshot_type _layer_stack;
	PVCore::PVCowValue<PVAxesCombination>::snapshot_type _axes_combination;
};
} // namespace Squey

#endif /* SQUEY_PVVIEWSTATE_H */
