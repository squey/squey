//! \file PVLayerFilter.cpp
//! $Id: PVLayerFilter.cpp 3199 2011-06-24 07:14:57Z aguinet $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011


#include <picviz/PVLayerFilter.h>
#include <assert.h>

/******************************************************************************
 *
 * Picviz::PVLayerFilter::PVLayerFilter
 *
 *****************************************************************************/
Picviz::PVLayerFilter::PVLayerFilter(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(Picviz::PVLayerFilter, args);
	set_output(NULL);
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilter)
{
	return PVCore::PVArgumentList();
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::operator_differentout
 *
 *****************************************************************************/
Picviz::PVLayer& Picviz::PVLayerFilter::operator_differentout(PVLayer &layer)
{
	operator()(layer, *_out_p);
	return *_out_p;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::operator()
 *
 *****************************************************************************/
Picviz::PVLayer& Picviz::PVLayerFilter::operator()(PVLayer& layer)
{
	assert(_view);
	if (_out_p)
		return operator_differentout(layer);
	return operator_sameout(layer);
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilter::operator()(PVLayer &in, PVLayer &out)
{
	// By default, if out != in, copy it
	if (&out != &in)
		out = in;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::operator_sameout
 *
 *****************************************************************************/
Picviz::PVLayer& Picviz::PVLayerFilter::operator_sameout(PVLayer &layer)
{
	operator()(layer, layer);
	return layer;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::set_output
 *
 *****************************************************************************/
void Picviz::PVLayerFilter::set_output(PVLayer* out)
{
	_out_p = out;

/*	AG: TOFIX: do not remember the clean way to do this... later
 	if (out == NULL)
		this->operator()(PVLayer&) = &(this->operator_sameout);
	else
		this->operator()(PVLayer&) = &(this->operator_differentout);*/
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::set_view
 *
 *****************************************************************************/
void Picviz::PVLayerFilter::set_view(PVView_p view)
{
	_view = view;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilter::get_default_args_for_view(PVView const& /*view*/)
{
	return get_args();
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::status_bar_description
 *
 *****************************************************************************/
QString Picviz::PVLayerFilter::status_bar_description()
{
	return QString();
}

/******************************************************************************
 *
 * Picviz::PVLayerFilter::detailed_description
 *
 *****************************************************************************/
QString Picviz::PVLayerFilter::detailed_description()
{
	return QString();
}

IMPL_FILTER(Picviz::PVLayerFilter)
