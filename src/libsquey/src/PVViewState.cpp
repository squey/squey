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

#include <squey/PVViewState.h>

void Squey::PVViewState::cool()
{
	if (is_cool() || _selection == nullptr) {
		return;
	}

	/* Held by somebody else -- the views still showing it, or another state
	 * pointing at the same one. Folding it would add a copy rather than
	 * replace one.
	 */
	if (_selection.use_count() > 1) {
		return;
	}

	PVCore::PVCompressedSelection folded = PVCore::PVCompressedSelection::from(*_selection);
	if (folded.is_empty()) {
		// It does not fold: a search left it scattered, and there is nothing
		// shorter to say than what it already says.
		return;
	}

	_folded = std::make_shared<const PVCore::PVCompressedSelection>(std::move(folded));
	_selection.reset();
}

PVCore::PVCowValue<Squey::PVSelection>::snapshot_type Squey::PVViewState::selection() const
{
	if (not is_cool()) {
		return _selection;
	}

	return std::make_shared<const PVSelection>(_folded->expand());
}
