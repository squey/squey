//! \file PVFieldFilterRegexpGrep.cpp
//! $Id: PVFieldFilterRegexpGrep.cpp 3100 2011-06-10 08:19:40Z aguinet $
//! Copyright (C) Sébastien Tricaud 2011-2011
//! Copyright (C) Philippe Saadé 2011-2011
//! Copyright (C) Picviz Labs 2011

#include "PVFieldFilterRegexpGrep.h"


/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrepg::PVFieldFilterGrep
 *
 *****************************************************************************/
PVFilter::PVFieldFilterRegexpGrep::PVFieldFilterRegexpGrep(PVCore::PVArgumentList const& args) :
	PVFilter::PVFieldsFilter<PVFilter::one_to_one>()
{
	INIT_FILTER(PVFilter::PVFieldFilterRegexpGrep, args);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(PVFilter::PVFieldFilterRegexpGrep)
{
	PVCore::PVArgumentList args;
	args["regexp"] = QRegExp("");
	args["reverse"] = false;
	return args;
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::set_args
 *
 *****************************************************************************/
void PVFilter::PVFieldFilterRegexpGrep::set_args(PVCore::PVArgumentList const& args)
{
	FilterT::set_args(args);
	_rx = _args["regexp"].toRegExp();
	_inverse = args["reverse"].toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterRegexpGrep::one_to_one(PVCore::PVField& obj)
{
	obj.init_qstr();
	QRegExp rx(_rx); // Local object (local to a thread !)
	bool found = (rx.indexIn(obj.qstr()) != -1);
	if (!(found ^ _inverse))
	{
		obj.set_invalid();
		// Invalidate the parent element
		obj.elt_parent()->set_invalid();
	}
	return obj;
}

IMPL_FILTER(PVFilter::PVFieldFilterRegexpGrep)
