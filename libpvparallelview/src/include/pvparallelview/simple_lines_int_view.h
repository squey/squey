/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
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
