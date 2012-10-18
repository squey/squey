/**
 * \file simple_lines_int_view.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef SLLINES_INT_H
#define SLLINES_INT_H

#include <pvparallelview/simple_lines_view.h>

class SLIntView: public SLView<int>
{
public:
	SLIntView(QWidget* parent): SLView<int>(parent) { }
private:
	void paintGL();
};

#endif
