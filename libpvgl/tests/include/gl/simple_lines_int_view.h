#ifndef SLLINES_INT_H
#define SLLINES_INT_H

#include <gl/simple_lines_view.h>

class SLIntView: public SLView<int>
{
public:
	SLIntView(QWidget* parent): SLView<int>(parent) { }
private:
	void paintGL();
};

#endif
