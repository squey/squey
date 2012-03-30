#ifndef PICVIZ_PVAD2BWIDGET_H
#define PICVIZ_PVAD2BWIDGET_H

#include <pvkernel/core/general.h>

namespace Picviz {

// Forward decalartion
class PVAD2B;

class LibPicvizExport PVAD2BWidget
{
public:
	PVAD2BWidget(PVAD2B& ad2b, QWidget* parent = NULL);
private:
	PVAD2B& _ad2b;
};

}

#endif
