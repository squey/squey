//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <squey/PVLayerFilter.h>
#include <squey/PVLayer.h>

#include <boost/thread.hpp>

#include <cassert>

/******************************************************************************
 *
 * Squey::PVLayerFilter::PVLayerFilter
 *
 *****************************************************************************/
Squey::PVLayerFilter::PVLayerFilter(PVCore::PVArgumentList const& args)
{
	INIT_FILTER(Squey::PVLayerFilter, args);
	set_output(nullptr);
	_should_cancel = false;
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Squey::PVLayerFilter)
{
	return {};
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::operator()
 *
 *****************************************************************************/
Squey::PVLayer const& Squey::PVLayerFilter::operator()(PVLayer const& layer)
{
	assert(_view);
	if (_out_p) {
		operator()(layer, *_out_p);
		return *_out_p;
	}
	throw std::runtime_error("Can't apply filter on same layer.");
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::set_output
 *
 *****************************************************************************/
void Squey::PVLayerFilter::set_output(PVLayer* out)
{
	_out_p = out;
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Squey::PVLayerFilter::get_default_args_for_view(PVView const& /*view*/)
{
	return get_args();
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::status_bar_description
 *
 *****************************************************************************/
QString Squey::PVLayerFilter::status_bar_description()
{
	return {};
}

/******************************************************************************
 *
 * Squey::PVLayerFilter::detailed_description
 *
 *****************************************************************************/
QString Squey::PVLayerFilter::detailed_description()
{
	return {};
}

Squey::PVLayerFilter::hash_menu_function_t const& Squey::PVLayerFilter::get_menu_entries() const
{
	return _menu_entries;
}

void Squey::PVLayerFilter::add_ctxt_menu_entry(QString menu_entry, ctxt_menu_f f)
{
	_menu_entries[menu_entry] = f;
}

void Squey::PVLayerFilter::cancel()
{
	_should_cancel = true;
}

bool Squey::PVLayerFilter::should_cancel()
{
	return _should_cancel;
}

PVCore::PVPluginPresets<Squey::PVLayerFilter> Squey::PVLayerFilter::get_presets()
{
	return PVCore::PVPluginPresets<PVLayerFilter>(*this, "presets/layer_filters");
}
