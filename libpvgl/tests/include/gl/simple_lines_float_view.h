/**
 * \file simple_lines_float_view.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef SLLINES_FLOAT_H
#define SLLINES_FLOAT_H

#include <gl/simple_lines_view.h>

class SLFloatView: public SLView<float>
{
public:
	SLFloatView(QWidget* parent): SLView<float>(parent) { }
private:
	void paintGL();
};

#endif
