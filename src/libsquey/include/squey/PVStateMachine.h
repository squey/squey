/* * MIT License
 *
 * © ESI Group, 2015
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

#ifndef SQUEY_PVSTATEMACHINE_H
#define SQUEY_PVSTATEMACHINE_H

#include <QString>

namespace Squey
{

class PVStateMachine
{
	/* lines states: this must *NOT* be handled by an enumeration
	 * as it will limits the different possibilities (enum max elements)
	 * we have with lines in the future and makes more complex the way
	 * we know about the given state we are.
	 */
	bool listing_unselected_visible;
	bool listing_zombie_visible;
	bool view_unselected_zombie_visible = true;

  public:
	PVStateMachine();

	// listing state management

	bool are_listing_all() const { return listing_unselected_visible && listing_zombie_visible; }

	bool are_listing_no_nu_nz() const
	{
		return !(listing_unselected_visible || listing_zombie_visible);
	}

	bool are_listing_no_nz() const { return listing_unselected_visible && !listing_zombie_visible; }

	bool are_listing_no_nu() const { return !listing_unselected_visible && listing_zombie_visible; }

	bool are_listing_unselected_visible() const { return listing_unselected_visible; }

	bool are_listing_zombie_visible() const { return listing_zombie_visible; }

	bool& are_view_unselected_zombie_visible() { return view_unselected_zombie_visible; }

	QString get_string();

	void set_listing_unselected_visible(bool visible) { listing_unselected_visible = visible; }

	void set_listing_zombie_visible(bool visible) { listing_zombie_visible = visible; }

	void set_view_unselected_zombie_visible(bool visible)
	{
		view_unselected_zombie_visible = visible;
	}

	void toggle_listing_unselected_visibility()
	{
		listing_unselected_visible = !listing_unselected_visible;
	}

	void toggle_listing_zombie_visibility() { listing_zombie_visible = !listing_zombie_visible; }

	void toggle_view_unselected_zombie_visibility()
	{
		view_unselected_zombie_visible = !view_unselected_zombie_visible;
	}
};
} // namespace Squey

#endif /* SQUEY_PVSTATEMACHINE_H */
