#ifndef BCCBVIEW_H
#define BCCBVIEW_H

#include <code_bz/bcode_cb.h>
#include <gl/simple_lines_view.h>

class BCCBView: public SLView<int>
{
public:
	BCCBView(QWidget* parent): SLView<int>(parent) { }
	void set_bccb(BCodeCB cb) { _cb = cb; }
private:
	void paintGL();
	BCodeCB _cb;
};

#endif
