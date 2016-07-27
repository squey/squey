/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef INENDI_PVSTATEMACHINE_H
#define INENDI_PVSTATEMACHINE_H

#include <QString>

namespace Inendi
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
	/** Square Area Modes
	 * helps to identify the selection mode
	 * that is done with the square area
	 */
	enum SquareAreaModes {
		AREA_MODE_OFF, /**< No selection area */
		AREA_MODE_SET_WITH_VOLATILE,
		AREA_MODE_ADD_VOLATILE,
		AREA_MODE_SUBSTRACT_VOLATILE,
		AREA_MODE_INTERSECT_VOLATILE
	};

  private:
	SquareAreaModes square_area_mode;

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

	SquareAreaModes get_square_area_mode() const { return square_area_mode; }

	void set_listing_unselected_visible(bool visible) { listing_unselected_visible = visible; }

	void set_listing_zombie_visible(bool visible) { listing_zombie_visible = visible; }

	void set_view_unselected_zombie_visible(bool visible)
	{
		view_unselected_zombie_visible = visible;
	}

	void set_square_area_mode(SquareAreaModes mode) { square_area_mode = mode; }

	void toggle_listing_unselected_visibility()
	{
		listing_unselected_visible = !listing_unselected_visible;
	}

	void toggle_listing_zombie_visibility() { listing_zombie_visible = !listing_zombie_visible; }

	void toggle_gl_unselected_visibility() { gl_unselected_visible = !gl_unselected_visible; }

	void toggle_gl_zombie_visibility() { gl_zombie_visible = !gl_zombie_visible; }

	void toggle_view_unselected_zombie_visibility()
	{
		view_unselected_zombie_visible = !view_unselected_zombie_visible;
	}
};
}

#endif /* INENDI_PVSTATEMACHINE_H */
