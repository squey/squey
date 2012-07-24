/**
 * \file PVLayerFilterWebmailFinder.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include "PVLayerFilterWebmailFinder.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <picviz/PVView.h>

/******************************************************************************
 *
 * Picviz::PVLayerFilterWebmailFinder::PVLayerFilterWebmailFinder
 *
 *****************************************************************************/
Picviz::PVLayerFilterWebmailFinder::PVLayerFilterWebmailFinder(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterWebmailFinder, l);
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterWebmailFinder)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterWebmailFinder)
{
	PVCore::PVArgumentList args;
	// args["Regular expression"] = QRegExp("(.*)");
	args["Domain axes"].setValue(PVCore::PVAxesIndexType());
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterWebmailFinder::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterWebmailFinder::get_default_args_for_view(PVView const& view)
{
	PVCore::PVArgumentList args;
	args["Domain axes"].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("domain"))));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterWebmailFinder::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterWebmailFinder::operator()(PVLayer& in, PVLayer &out)
{	
	PVCore::PVAxesIndexType axes_id = _args["Domain axis"].value<PVCore::PVAxesIndexType>();

	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();

	PVSelection hotmail_sel;
	PVLinesProperties hotmail_lp;
	QString hotmail("mail.live.com");

	QString yahoo("mail.yahoo.com");
	PVSelection yahoo_sel;
	PVLinesProperties yahoo_lp;

	for (unsigned int i = 0; i < axes_id.size(); i++) {
		int axis_id = axes_id[i];
		for (PVRow r = 0; r < nb_lines; r++) {
			if (should_cancel()) {
				if (&in != &out) {
					out = in;
				}
				return;
			}
			if (_view->get_line_state_in_pre_filter_layer(r)) {
				QString data = nraw.at(r, axis_id).get_qstr();
				hotmail_sel.set_line(r, data.contains(hotmail, Qt::CaseInsensitive));
				yahoo_sel.set_line(r, data.contains(yahoo, Qt::CaseInsensitive));
			}
		}
	}

	PVLayer yahoo_layer("Yahoo", yahoo_sel, yahoo_lp);
	_view->layer_stack.append_layer(yahoo_layer);

	PVLayer hotmail_layer("Hotmail", hotmail_sel, hotmail_lp);
	_view->layer_stack.append_layer(hotmail_layer);
}

IMPL_FILTER(Picviz::PVLayerFilterWebmailFinder)
