//! \file PVLayerFilterCreateLayers.cpp
//! $Id$
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QFile>
#include <QString>
#include <QByteArray>

#include "PVLayerFilterCreateLayers.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <picviz/PVView.h>

int Picviz::PVLayerFilterCreateLayers::create_layers_parse_config(QString filename)
{

	// PVLOG_INFO("Parse config for %s\n", filename);
	QRegExp re_section("^\\[(.*)\\]$", Qt::CaseSensitive, QRegExp::RegExp);
	QString section_name;
	int line_pos = 1;

	QFile configfile(filename);
	if (!configfile.open(QIODevice::ReadOnly | QIODevice::Text)) {
		return -1;
	}

	while (!configfile.atEnd()) {
		QByteArray line = configfile.readLine();
		line.chop(1);
		if (!line.isEmpty()) {
			int ret = re_section.indexIn(line);
			if (ret >= 0) {
			        section_name = re_section.cap(1);
				// PVLOG_INFO("Section:%s\n", qPrintable(section_name));
			} else {
				if (section_name.isEmpty()) {
					PVLOG_ERROR("Create Layer config issue: empty section found at line %d\n", line_pos);
				} else {
				// We know in which section we are and we parse data

					
				}
			}
		}

		line_pos++;
		// process_line(line);
	}
	return 0;
}


/******************************************************************************
 *
 * Picviz::PVLayerFilterCreateLayers::PVLayerFilterCreateLayers
 *
 *****************************************************************************/
Picviz::PVLayerFilterCreateLayers::PVLayerFilterCreateLayers(PVCore::PVArgumentList const& l)
	: PVLayerFilter(l)
{
	INIT_FILTER(PVLayerFilterCreateLayers, l);
	create_layers_parse_config(QString("create-layers.conf"));
}

/******************************************************************************
 *
 * DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterCreateLayers)
 *
 *****************************************************************************/
DEFAULT_ARGS_FILTER(Picviz::PVLayerFilterCreateLayers)
{
	PVCore::PVArgumentList args;
	// args["Regular expression"] = QRegExp("(.*)");
	args["Domain axes"].setValue(PVCore::PVAxesIndexType());
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterCreateLayers::get_default_args_for_view
 *
 *****************************************************************************/
PVCore::PVArgumentList Picviz::PVLayerFilterCreateLayers::get_default_args_for_view(PVView const& view)
{
	PVCore::PVArgumentList args;
	args["Domain axes"].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("domain"))));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterCreateLayers::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterCreateLayers::operator()(PVLayer& in, PVLayer &out)
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
				QString const& data = nraw.at(r, axis_id).get_qstr();
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

IMPL_FILTER(Picviz::PVLayerFilterCreateLayers)
