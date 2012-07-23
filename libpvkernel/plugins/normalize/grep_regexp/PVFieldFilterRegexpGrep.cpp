/**
 * \file PVFieldFilterRegexpGrep.cpp
 *
 * Copyright (C) Picviz Labs 2011-2012
 */

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
	args["regexp"] = QString("");
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
	_rx = QRegExp(args["regexp"].toString());
	_inverse = args["reverse"].toBool();
}

/******************************************************************************
 *
 * PVFilter::PVFieldFilterRegexpGrep::one_to_one
 *
 *****************************************************************************/
PVCore::PVField& PVFilter::PVFieldFilterRegexpGrep::one_to_one(PVCore::PVField& obj)
{
	QRegExp rx(_rx); // Local object (local to a thread !)
	QString str_tmp;
	bool found = (rx.indexIn(obj.get_qstr(str_tmp)) != -1);
	if (!(found ^ _inverse))
	{
		obj.set_invalid();
		// Invalidate the parent element
		obj.elt_parent()->set_invalid();
	}
	return obj;
}

IMPL_FILTER(PVFilter::PVFieldFilterRegexpGrep)
