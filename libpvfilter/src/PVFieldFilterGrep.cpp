//! \file PVFieldFilterGrep.cpp
//! $Id: PVFieldFilterGrep.cpp 3221 2011-06-30 11:45:19Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include <pvcore/PVField.h>
#include <pvfilter/PVFieldFilterGrep.h>


/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::PVFieldFilterGrep
 *
 *****************************************************************************/
PVFilter::PVFieldFilterGrep::PVFieldFilterGrep(PVArgumentList const& args) :
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>()
{
	INIT_FILTER(PVFilter::PVFieldFilterGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterGrep)
{
	PVArgumentList args;
	args["str"] = QString();
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldFilterGrep::set_args(PVArgumentList const& args)
{
	FilterT::set_args(args);
	_str = _args["str"].toString();
	_inverse = args["reverse"].toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterGrep::one_to_one(PVCore::PVField& obj)
{
	QString str = QString::fromAscii(obj.begin(), obj.size());
	bool found = str.contains(_str);
	if (!(found ^ _inverse)) {
		obj.set_invalid();
		obj.elt_parent()->set_invalid();
	}
	return obj;
}

IMPL_FILTER(PVFilter::PVFieldFilterGrep)
