/**
 * \file PVAxisComputationPython.h
 *
 * Copyright (C) Picviz Labs 2010-2012
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
