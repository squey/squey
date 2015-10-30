/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef SLLINES_FLOAT_H
#define SLLINES_FLOAT_H

#include <pvparallelview/simple_lines_view.h>

class SLFloatView: public SLView<float>
{
public:
	SLFloatView(QWidget* parent): SLView<float>(parent) { }
private:
	void paintGL();
};

#endif
