/**
 * @file
 *
 * @copyright (C) Picviz Labs 2009-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <inendi/PVLayerFilter.h>
#include <inendi/PVLayer.h>

#include <boost/bind.hpp>
#include <boost/thread.hpp>

#include <assert.h>

/******************************************************************************
 *
 * Inendi::PVLayerFilter::PVLayerFilter
 *
 *****************************************************************************/
Inendi::PVLayerFilter::PVLayerFilter(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(Inendi::PVLayerFilter, args);
	set_output(NULL);
	_should_cancel = false;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Inendi::PVLayerFilter)
{
	return PVCore::PVArgumentList();
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::operator_differentout
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVLayerFilter::operator_differentout(PVLayer &layer)
{
	operator()(layer, *_out_p);
	return *_out_p;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::operator()
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVLayerFilter::operator()(PVLayer& layer)
{
	assert(_view);
	if (_out_p)
		return operator_differentout(layer);
	return operator_sameout(layer);
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::operator()
 *
 *****************************************************************************/
void Inendi::PVLayerFilter::operator()(PVLayer &in, PVLayer &out)
{
	// By default, if out != in, copy it
	if (&out != &in)
		out = in;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::operator_sameout
 *
 *****************************************************************************/
Inendi::PVLayer& Inendi::PVLayerFilter::operator_sameout(PVLayer &layer)
{
	operator()(layer, layer);
	return layer;
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::set_output
 *
 *****************************************************************************/
void Inendi::PVLayerFilter::set_output(PVLayer* out)
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
 * Inendi::PVLayerFilter::set_view
 *
 *****************************************************************************/
void Inendi::PVLayerFilter::set_view(PVView_sp const& view)
{
	_view = view.get();
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Inendi::PVLayerFilter::get_default_args_for_view(PVView const& /*view*/)
{
	return get_args();
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::status_bar_description
 *
 *****************************************************************************/
QString Inendi::PVLayerFilter::status_bar_description()
{
	return QString();
}

/******************************************************************************
 *
 * Inendi::PVLayerFilter::detailed_description
 *
 *****************************************************************************/
QString Inendi::PVLayerFilter::detailed_description()
{
	return QString();
}

Inendi::PVLayerFilter::hash_menu_function_t const& Inendi::PVLayerFilter::get_menu_entries() const
{
	return _menu_entries;
}

void Inendi::PVLayerFilter::add_ctxt_menu_entry(QString menu_entry, ctxt_menu_f f)
{
	_menu_entries[menu_entry] = f;
}

boost::thread Inendi::PVLayerFilter::launch_in_thread(PVLayer& layer)
{
	return boost::thread(boost::bind(&PVLayerFilter::operator(), this, layer));
}

void Inendi::PVLayerFilter::cancel()
{
	_should_cancel = true;
}

bool Inendi::PVLayerFilter::should_cancel()
{
	return _should_cancel;
}

PVCore::PVTag<Inendi::PVLayerFilter> Inendi::PVLayerFilter::get_tag(QString const& name)
{
	return LIB_CLASS(PVLayerFilter)::get().get_tag(name);
}

PVCore::PVPluginPresets<Inendi::PVLayerFilter> Inendi::PVLayerFilter::get_presets()
{
	return PVCore::PVPluginPresets<PVLayerFilter>(*this, "presets/layer_filters");
}

IMPL_FILTER(Inendi::PVLayerFilter)
