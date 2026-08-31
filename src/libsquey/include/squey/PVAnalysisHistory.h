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

#ifndef SQUEY_PVANALYSISHISTORY_H
#define SQUEY_PVANALYSISHISTORY_H

#include <squey/PVViewState.h>

#include <sigc++/sigc++.h>

#include <QString>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

namespace Squey
{

class PVRoot;
class PVView;

/**
 * \class PVAnalysisStep
 *
 * One step of an analysis: what every view looked like once it was done, and
 * enough about it to be named in a breadcrumb.
 *
 * A step remembers the state the views were left in, not the operation that
 * left them there. That is what lets the history be walked in any order --
 * clicking the third crumb lands on the third state directly, rather than
 * having to undo its way back to it.
 */
class PVAnalysisStep
{
	friend class PVAnalysisHistory;

  public:
	/**
	 * What the step is called. Shown as a tooltip rather than as text: a
	 * breadcrumb has to stay narrow enough to sit under a toolbar.
	 */
	QString const& label() const { return _label; }

	/**
	 * Which icon stands for the step, named as PVModdedIcon names them. This
	 * is what the breadcrumb actually shows, so it should say what kind of act
	 * the step was -- a search, a gradient, a layer -- rather than which one.
	 */
	std::string const& icon() const { return _icon; }

	/**
	 * How many rows the step left selected in the view it was taken on, which
	 * is the one thing about a step worth showing next to its name.
	 */
	size_t selected_row_count() const { return _selected_row_count; }

	/**
	 * Whether this step is the one the views are currently showing.
	 */
	bool is_empty() const { return _states.empty(); }

  private:
	QString _label;
	std::string _icon;
	std::string _merge_key;
	std::chrono::steady_clock::time_point _taken_at;
	size_t _selected_row_count = 0;
	std::vector<std::pair<PVView*, PVViewState>> _states;
};

/**
 * \class PVAnalysisHistory
 *
 * The steps an analysis went through, and where in them the user currently
 * stands. One per investigation rather than one per view, because a single
 * gesture can move more than one view: a correlation propagates a selection
 * from the view it was made in to another, possibly under another source. Two
 * histories would then disagree about what undoing means.
 *
 * Steps are delimited by PVAnalysisHistory::Scope rather than recorded per
 * mutation, because one thing the user did is many things the code did:
 * creating a layer from a set of values runs a search, adds a layer, toggles
 * its visibility, commits a selection into it and moves it, and all of that is
 * one crumb.
 *
 * A scope records a step when a value was written to, not when the bits it
 * holds came out different. So a dialog opened and cancelled leaves nothing --
 * nothing was written -- while emptying an already empty selection does leave a
 * crumb. Telling those two apart would mean comparing the contents of every
 * layer of every view on each action, a hundred megabytes a step on a large
 * collection, to spare the breadcrumb an entry for something the user did ask
 * for.
 */
class PVAnalysisHistory
{
  public:
	/**
	 * How long after a step another one carrying the same key still joins it.
	 * Dragging a selection rectangle commits every 300 ms, and a drag is one
	 * step rather than one per commit.
	 */
	static constexpr std::chrono::milliseconds default_merge_window{800};

	/**
	 * How many steps are kept. Old ones fall off the far end, because what a
	 * step holds alone is freed only once no step holds it.
	 */
	static constexpr size_t default_max_steps = 64;

  public:
	explicit PVAnalysisHistory(PVRoot& root) : _root(root) {}

  public:
	/**
	 * \class Scope
	 *
	 * Delimits one step. Whatever happens while it is alive is one crumb, and
	 * nothing is recorded if nothing changed.
	 *
	 * Scopes nest: an operation built out of smaller ones that each open their
	 * own scope still yields a single step, named by the outermost.
	 */
	class Scope
	{
	  public:
		Scope(PVRoot& root, QString label, std::string icon);

		/**
		 * As above, but joining the step before it when that one carries the
		 * same key and is recent enough. Pass a key per kind of gesture, so
		 * that a rectangle being dragged does not merge with a menu action
		 * that happened to follow it.
		 */
		Scope(PVRoot& root,
		      QString label,
		      std::string icon,
		      std::string merge_key,
		      std::chrono::milliseconds window = default_merge_window);

		/**
		 * As above, reaching the history through the view being worked on,
		 * which is what most call sites have to hand.
		 */
		Scope(PVView& view, QString label, std::string icon);
		Scope(PVView& view,
		      QString label,
		      std::string icon,
		      std::string merge_key,
		      std::chrono::milliseconds window = default_merge_window);

		~Scope();

		Scope(Scope const&) = delete;
		Scope& operator=(Scope const&) = delete;

	  private:
		PVAnalysisHistory& _history;
		QString _label;
		std::string _icon;
		std::string _merge_key;
		std::chrono::milliseconds _window;
	};

  public:
	size_t size() const { return _steps.size(); }
	bool is_empty() const { return _steps.empty(); }

	/**
	 * Which step the views are showing, counted from the oldest.
	 */
	size_t position() const { return _position; }

	PVAnalysisStep const& step(size_t index) const { return _steps[index]; }

	bool can_undo() const { return _position > 0; }
	bool can_redo() const { return not _steps.empty() && _position + 1 < _steps.size(); }

	void undo();
	void redo();

	/**
	 * Lands on any step directly, which is what a breadcrumb is for.
	 */
	void go_to(size_t index);

	/**
	 * Drops everything, for when something happened that cannot be undone --
	 * deleting a column erases it from disk, so no earlier step could be
	 * restored without naming an axis whose data is gone.
	 */
	void clear();

	/**
	 * Forgets a view that is being deleted, so that no step keeps pointing at
	 * it. Steps left holding nothing go too.
	 */
	void forget(PVView* view);

	/**
	 * How many steps are kept before the oldest starts falling off.
	 */
	void set_max_steps(size_t max_steps);
	size_t max_steps() const { return _max_steps; }

  public:
	/**
	 * Emitted whenever the steps or the position change, so that a breadcrumb
	 * can follow along.
	 */
	sigc::signal<void()> _changed;

  private:
	void open();
	void close(QString label,
	           std::string icon,
	           std::string merge_key,
	           std::chrono::milliseconds window);

	PVAnalysisStep capture(QString label, std::string icon, std::string merge_key) const;
	static bool same_states(PVAnalysisStep const& a, PVAnalysisStep const& b);
	void restore(PVAnalysisStep const& step);
	void drop_oldest_steps();

  private:
	PVRoot& _root;
	std::vector<PVAnalysisStep> _steps;
	size_t _position = 0;
	size_t _max_steps = default_max_steps;
	unsigned _depth = 0;
	/* Restoring a step mutates the views, and those mutations must not be read
	 * back as a step of their own.
	 */
	bool _restoring = false;
};
} // namespace Squey

#endif /* SQUEY_PVANALYSISHISTORY_H */
