/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVAXISCOMPUTATIONPYTHON_H
#define PVAXISCOMPUTATIONPYTHON_H

#include <picviz/PVAxisComputation.h>

namespace Picviz {

class PVAxisComputationPython: public PVAxisComputation
{
public:
	PVAxisComputationPython(PVCore::PVArgumentList const& args = PVAxisComputationPython::default_args());

public:
	bool operator()(PVRush::PVNraw* nraw);
	QString get_human_name() const { return QString("Python"); }

	CLASS_FILTER(PVAxisComputationPython)
};

}

#endif
