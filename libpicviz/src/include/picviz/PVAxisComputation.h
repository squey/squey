/**
 * \file PVAxisComputation.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef PICVIZ_PVAXISCOMPITATION_H
#define PICVIZ_PVAXISCOMPITATION_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/filter/PVFilterFunction.h>

#include <boost/shared_ptr.hpp>

namespace PVRush {
class PVNraw;
}

namespace Picviz {

class PVSource;

class PVAxisComputation: public PVFilter::PVFilterFunctionBase<bool, PVRush::PVNraw*>, public PVCore::PVRegistrableClass<PVAxisComputation>
{
public:
	typedef boost::shared_ptr<PVAxisComputation> p_type;
public:
	virtual ~PVAxisComputation() { }
	virtual QString get_human_name() const = 0;
};

typedef PVAxisComputation::func_type PVAxisComputation_f;
typedef PVAxisComputation::p_type PVAxisComputation_p;

}

#endif
