/**
 * \file PVLayerFilterCreateLayers.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QFile>
#include <QString>
#include <QByteArray>

#include "parse-config.h"
#include "PVLayerFilterCreateLayers.h"
#include <pvkernel/core/PVColor.h>
#include <pvkernel/core/PVAxesIndexType.h>
#include <picviz/PVView.h>


/******************************************************************************
 *
 * Picviz::PVLayerFilterCreateLayers::PVLayerFilterCreateLayers
 *
 *****************************************************************************/
Picviz::PVLayerFilterCreateLayers::PVLayerFilterCreateLayers(QString section_name, QMap<QString, QStringList> layers_regex, PVCore::PVArgumentList const& l)
	: PVLayerFilter(l),
	  _section_name(section_name),
	  _layers_regex(layers_regex)
{
	INIT_FILTER(PVLayerFilterCreateLayers, l);
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
	args["Search axis"].setValue(PVCore::PVAxesIndexType());
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
	args["Search axis"].setValue(PVCore::PVAxesIndexType(view.get_original_axes_index_with_tag(get_tag("domain"))));
	return args;
}

/******************************************************************************
 *
 * Picviz::PVLayerFilterCreateLayers::operator()
 *
 *****************************************************************************/
void Picviz::PVLayerFilterCreateLayers::operator()(PVLayer& in, PVLayer &out)
{	
	PVCore::PVAxesIndexType axes_id = _args["Search axis"].value<PVCore::PVAxesIndexType>();

	PVRow nb_lines = _view->get_qtnraw_parent().get_nrows();

	PVRush::PVNraw::nraw_table const& nraw = _view->get_qtnraw_parent();


	PVLinesProperties generic_lp;

	QMapIterator<QString, QStringList> layers_to_create(_layers_regex);
	while(layers_to_create.hasNext()) {
		layers_to_create.next();

		// We compile all the required regex so we can run a fast search
		QList<QRegExp> layers_compiled_regex;
		for (int lr = 0; lr < layers_to_create.value().size(); lr++) {
			PVLOG_INFO("Compile regex:%s\n",qPrintable(layers_to_create.value().at(lr)));
			layers_compiled_regex.append(QRegExp(layers_to_create.value().at(lr)));
		}


		PVSelection layer_selection;
		bool this_layer_has_a_selection = false;

		for (unsigned int i = 0; i < axes_id.size(); i++) {
			int axis_id = axes_id[i];
			// Check if we shall cancel stuff
			for (PVRow r = 0; r < nb_lines; r++) {
				if (should_cancel()) {
					if (&in != &out) {
						out = in;
					}
					return;
				}
				if (_view->get_line_state_in_pre_filter_layer(r)) {
					// I run my regex on the data to create the layers
					for (int layer_regex_i = 0; layer_regex_i < layers_compiled_regex.size(); layer_regex_i++) {
						QString data = nraw.at(r, axis_id).get_qstr();	
						int sel = layers_compiled_regex[layer_regex_i].indexIn(data);
						if (sel == 0) { this_layer_has_a_selection = true; }
						layer_selection.set_line_select_only(r, !sel);
					}
				}
			}
		}

		// I really create my new layer
		if (this_layer_has_a_selection) {
			PVLayer new_layer(layers_to_create.key(), layer_selection, generic_lp);
			_view->layer_stack.append_layer(new_layer);
		}

	}

}

IMPL_FILTER(Picviz::PVLayerFilterCreateLayers)
