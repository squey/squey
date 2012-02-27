//! \file PVIdleManager.h
//! $Id: PVIdleManager.h 2975 2011-05-26 03:54:42Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saade 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef LIBPVGL_PVIDLE_MANAGER_H
#define LIBPVGL_PVIDLE_MANAGER_H

#include <pvkernel/core/general.h>
#include <tbb/tick_count.h>

namespace PVGL {
class PVDrawable;

/**
* \enum PVIdleTaskKinds
*/
enum PVIdleTaskKinds {
	IDLE_REDRAW_LINES,
	IDLE_REDRAW_ZOMBIE_LINES,
	IDLE_REDRAW_MAP_LINES,
	IDLE_REDRAW_ZOMBIE_MAP_LINES,
};

/**
 *
 */
class LibGLDecl PVIdleManager {

	struct IdleTask {
		PVGL::PVDrawable      *drawable; //!<
		PVGL::PVIdleTaskKinds  kind;     //!<

		/**
		*  Constructor.
		*/
		IdleTask(PVGL::PVDrawable *drawable_, PVGL::PVIdleTaskKinds kind_):drawable(drawable_),kind(kind_){}

		/**
		*
		*/
		bool operator==(const IdleTask&t)const
			{
				return drawable == t.drawable && kind == t.kind;
			}

		/**
		*
		*/
		bool operator<(const IdleTask&t)const
			{
				if (drawable > t.drawable || (drawable == t.drawable && kind > t.kind)) {
					return true;
				}
				return false;
			}
	};

	struct IdleValue {
		int  nb_lines;  //!<
		bool removed;   //!<
		tbb::tick_count time_start;
		IdleValue(int nb_lines_ = 25000, bool removed_ = false):nb_lines(nb_lines_),removed(removed_){ time_start = tbb::tick_count::now(); }
	};

	std::map<IdleTask, IdleValue> tasks; //!<
public:
	/**
	*
	* @param drawable
	* @param kind
	*
	* @return
	*/
	bool task_exists(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind) const;

	/**
	*
	*/
	void callback();

	/**
	* @param drawable
	* @param kind
	*/
	void new_task(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind);

	/**
	*
	* @param drawable
	* @param kind
	*
	* @return
	*/
	int get_number_of_lines(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind);

	/**
	*
	* @param drawable
	* @param kind
	*/
	void remove_task(PVGL::PVDrawable *drawable, PVGL::PVIdleTaskKinds kind);

	/**
	*
	* @param drawable
	*/
	void remove_tasks_for_drawable(PVGL::PVDrawable *drawable);
};
}
#endif
