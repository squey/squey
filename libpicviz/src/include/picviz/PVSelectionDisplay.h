//! \file PVSelectionDisplay
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#ifndef PICVIZ_PVSELECTIONDISPLAY_H
#define PICVIZ_PVSELECTIONDISPLAY_H

#include <picviz/general.h>

namespace Picviz {

	class LibExport PVSelectionDisplay {
	public:
		/** Selection Display Mode
		 *  It is possible to choose the wanted events to see
		 *  - All types of lines
		 *  - No unselected lines
		 *  - No zombies (in a layer)
		 *  - No unselected AND no zombies
		 */
		typedef enum {
			ALL           = 0x01, /**< Display all lines */
			NO_UNSELECTED = 0x02, /**< Don't display unselected lines */
			NO_ZOMBIES    = 0x04, /**< Don't display zombies lines */
		} PVSelectionDisplayMode_t;
	};

}

#endif	/* PICVIZ_PVSELECTIONDISPLAY_H */
