/**
 * \file PVSquareArea.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PICVIZ_PVSQUAREAREA_H
#define PICVIZ_PVSQUAREAREA_H

#include <pvkernel/core/general.h>

namespace Picviz {

/**
 * \class PVSquareArea
 */
class LibPicvizDecl PVSquareArea {
public:
	PVSquareArea();

public:
	float get_end_x() const;
	float get_end_y() const;
	float get_start_x() const;
	float get_start_y() const;


	void set_end(float ex, float ey);
	void set_end_x(float ex);
	void set_end_y(float ey);
	void set_start(float sx, float sy);
	void set_start_x(float sx);
	void set_start_y(float sy);

	bool is_dirty() const;
	void set_dirty();
	void set_clean();
	bool is_empty() const;
private:
	float end_x;
	float end_y;
	float start_x;
	float start_y;
	bool  dirty;
};

}

#endif
