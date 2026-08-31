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
#include <squey/PVSelection.h>
#include <squey/PVView.h>

#include <algorithm>
#include <list>
#include <utility>

/******************************************************************************
 * Squey::PVAnalysisHistory::Scope
 *****************************************************************************/

Squey::PVAnalysisHistory::Scope::Scope(PVRoot& root, QString label, std::string icon)
    : Scope(root, std::move(label), std::move(icon), std::string(), default_merge_window)
{
}

Squey::PVAnalysisHistory::Scope::Scope(PVRoot& root,
                                       QString label,
                                       std::string icon,
                                       std::string merge_key,
                                       std::chrono::milliseconds window)
    : _history(root.history())
{
	_history.open();

	/* Only the outermost scope names the step: an operation assembled out of
	 * smaller ones is one thing the user did, not several.
	 */
	_label = std::move(label);
	_icon = std::move(icon);
	_merge_key = std::move(merge_key);
	_window = window;
}

Squey::PVAnalysisHistory::Scope::Scope(PVView& view, QString label, std::string icon)
    : Scope(view.get_parent<PVRoot>(), std::move(label), std::move(icon))
{
}

Squey::PVAnalysisHistory::Scope::Scope(PVView& view,
                                       QString label,
                                       std::string icon,
                                       std::string merge_key,
                                       std::chrono::milliseconds window)
    : Scope(view.get_parent<PVRoot>(),
            std::move(label),
            std::move(icon),
            std::move(merge_key),
            window)
{
}

Squey::PVAnalysisHistory::Scope::~Scope()
{
	_history.close(std::move(_label), std::move(_icon), std::move(_merge_key), _window);
}

/******************************************************************************
 * Squey::PVAnalysisHistory::open
 *****************************************************************************/

void Squey::PVAnalysisHistory::open()
{
	if (_depth++ > 0 || _restoring) {
		return;
	}

	/* The history holds the state the views are in as its current step, so it
	 * needs one to start from before anything can be undone back to it.
	 */
	if (_steps.empty()) {
		_steps.push_back(capture(QString("Opened"), "folder-open", std::string()));
		_position = 0;
	}
}

/******************************************************************************
 * Squey::PVAnalysisHistory::close
 *****************************************************************************/

void Squey::PVAnalysisHistory::close(QString label,
                                     std::string icon,
                                     std::string merge_key,
                                     std::chrono::milliseconds window)
{
	if (--_depth > 0 || _restoring) {
		return;
	}

	PVAnalysisStep step = capture(std::move(label), std::move(icon), std::move(merge_key));

	/* A barrier may have wiped the history from inside the very scope being
	 * closed, in which case this step is the one everything else starts from.
	 */
	if (_steps.empty()) {
		_steps.push_back(std::move(step));
		_position = 0;
		_changed.emit();
		return;
	}

	/* Nothing changed, so there is nothing to come back to. Opening a dialog
	 * and cancelling it must not leave a crumb behind.
	 */
	if (same_states(step, _steps[_position])) {
		return;
	}

	/* Acting after having gone back drops whatever was ahead: the history is a
	 * line, so the branch that is not being followed is let go of.
	 */
	_steps.erase(_steps.begin() + long(_position) + 1, _steps.end());

	PVAnalysisStep const& previous = _steps[_position];
	const bool joins_previous = not step._merge_key.empty() &&
	                            step._merge_key == previous._merge_key &&
	                            step._taken_at - previous._taken_at <= window;

	if (joins_previous && _position > 0) {
		/* A drag is one step, however many times it committed on its way. The
		 * step it joins keeps its own name, since both describe the same
		 * gesture, but takes the newer state and the newer time -- so that a
		 * drag going on for a while keeps merging rather than falling out of
		 * the window against its first commit.
		 */
		step._label = previous._label;
		step._icon = previous._icon;
		_steps[_position] = std::move(step);
	} else {
		_steps.push_back(std::move(step));
		++_position;
		drop_oldest_steps();
	}

	_changed.emit();
}

/******************************************************************************
 * Squey::PVAnalysisHistory::capture
 *****************************************************************************/

Squey::PVAnalysisStep Squey::PVAnalysisHistory::capture(QString label,
                                                        std::string icon,
                                                        std::string merge_key) const
{
	PVAnalysisStep step;

	step._label = std::move(label);
	step._icon = std::move(icon);
	step._merge_key = std::move(merge_key);
	step._taken_at = std::chrono::steady_clock::now();

	for (PVView* view : _root.get_children<PVView>()) {
		step._states.emplace_back(view, view->capture_state());
	}

	if (PVView const* view = _root.current_view()) {
		step._selected_row_count = view->get_real_output_selection().bit_count();
	}

	for (auto const& [id, contributor] : _contributors) {
		if (PVAnalysisAttachment attachment = contributor.capture()) {
			step._attachments.emplace(id, std::move(attachment));
		}
	}

	return step;
}

/******************************************************************************
 * Squey::PVAnalysisHistory::same_states
 *****************************************************************************/

bool Squey::PVAnalysisHistory::same_states(PVAnalysisStep const& a, PVAnalysisStep const& b)
{
	if (a._states.size() != b._states.size()) {
		return false;
	}

	for (size_t i = 0; i < a._states.size(); i++) {
		if (a._states[i].first != b._states[i].first ||
		    not a._states[i].second.holds_same_as(b._states[i].second)) {
			return false;
		}
	}

	return true;
}

/******************************************************************************
 * Squey::PVAnalysisHistory::restore
 *****************************************************************************/

void Squey::PVAnalysisHistory::restore(PVAnalysisStep const& step)
{
	/* Putting the views back changes them, and those changes are the step being
	 * landed on rather than a step of their own.
	 */
	_restoring = true;

	const std::list<PVView*> alive = _root.get_children<PVView>();
	for (auto const& [view, state] : step._states) {
		if (std::find(alive.begin(), alive.end(), view) != alive.end()) {
			view->restore_state(state);
		}
	}

	/* After the states, not before: putting a selection back makes the views
	 * react, and one of the things they do is drop the rectangle that no longer
	 * describes what is selected. This hands it back once they have.
	 */
	for (auto const& [id, contributor] : _contributors) {
		const auto attachment = step._attachments.find(id);
		contributor.restore(attachment == step._attachments.end() ? PVAnalysisAttachment()
		                                                          : attachment->second);
	}

	_restoring = false;
}

/******************************************************************************
 * Squey::PVAnalysisHistory::undo / redo / go_to
 *****************************************************************************/

void Squey::PVAnalysisHistory::undo()
{
	if (can_undo()) {
		go_to(_position - 1);
	}
}

void Squey::PVAnalysisHistory::redo()
{
	if (can_redo()) {
		go_to(_position + 1);
	}
}

void Squey::PVAnalysisHistory::go_to(size_t index)
{
	if (index >= _steps.size() || index == _position) {
		return;
	}

	_position = index;
	restore(_steps[_position]);

	_changed.emit();
}

/******************************************************************************
 * Squey::PVAnalysisHistory::clear
 *****************************************************************************/

void Squey::PVAnalysisHistory::clear()
{
	if (_steps.empty()) {
		return;
	}

	_steps.clear();
	_position = 0;

	_changed.emit();
}

/******************************************************************************
 * Squey::PVAnalysisHistory::forget
 *****************************************************************************/

void Squey::PVAnalysisHistory::forget(PVView* view)
{
	bool changed = false;

	for (PVAnalysisStep& step : _steps) {
		const size_t before = step._states.size();
		std::erase_if(step._states, [view](auto const& s) { return s.first == view; });
		changed |= step._states.size() != before;
	}

	/* A step that ends up holding no view has nothing left to land on.
	 */
	for (size_t i = _steps.size(); i-- > 0;) {
		if (_steps[i]._states.empty()) {
			_steps.erase(_steps.begin() + long(i));
			if (_position > i) {
				--_position;
			} else if (_position == i && _position > 0) {
				--_position;
			}
			changed = true;
		}
	}

	if (_steps.empty()) {
		_position = 0;
	}

	if (changed) {
		_changed.emit();
	}
}

/******************************************************************************
 * Squey::PVAnalysisHistory::set_max_steps / drop_oldest_steps
 *****************************************************************************/

void Squey::PVAnalysisHistory::set_max_steps(size_t max_steps)
{
	_max_steps = std::max(size_t(1), max_steps);
	drop_oldest_steps();
}

void Squey::PVAnalysisHistory::drop_oldest_steps()
{
	if (_steps.size() <= _max_steps) {
		return;
	}

	const size_t excess = _steps.size() - _max_steps;
	_steps.erase(_steps.begin(), _steps.begin() + long(excess));
	_position = _position > excess ? _position - excess : 0;
}

/******************************************************************************
 * Squey::PVAnalysisHistory::add_contributor / remove_contributor
 *****************************************************************************/

Squey::PVAnalysisHistory::ContributorId Squey::PVAnalysisHistory::add_contributor(
    std::function<PVAnalysisAttachment()> capture,
    std::function<void(const PVAnalysisAttachment&)> restore)
{
	const ContributorId id = _next_contributor_id++;
	_contributors.emplace(id, Contributor{std::move(capture), std::move(restore)});

	return id;
}

void Squey::PVAnalysisHistory::remove_contributor(ContributorId id)
{
	_contributors.erase(id);

	/* What it had handed over is left where it is: an attachment nobody claims
	 * costs a pointer, and dropping it from every step would be work done for
	 * a contributor that is on its way out anyway.
	 */
}
