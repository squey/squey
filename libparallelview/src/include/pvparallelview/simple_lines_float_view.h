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
