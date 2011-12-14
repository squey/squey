#ifndef PVAXISCOMPUTATIONPYTHON_H
#define PVAXISCOMPUTATIONPYTHON_H

#include <picviz/PVAxisComputation.h>

namespace Picviz {

class PVAxisComputationPython: public PVAxisComputation
{
public:
	PVAxisComputationPython(PVCore::PVArgumentList const& args = PVAxisComputationPython::default_args());

public:
	bool operator()(PVSource* src);
	QString get_human_name() const { return QString("Python"); }

	CLASS_FILTER(PVAxisComputationPython)
};

}

#endif
