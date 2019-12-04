/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2019
 */

#include <pvbase/general.h>

#include "PVInputTypeERF.h"

bool PVRush::PVInputTypeERF::createWidget(hash_formats& formats,
                                          list_inputs& inputs,
                                          QString& format,
                                          PVCore::PVArgumentList& /*args_ext*/,
                                          QWidget* parent) const
{
	inputs.push_back(PVInputDescription_p(new PVERFDescription(
	    "/srv/logs/VW/BOOST_fill_sol_V01_OPT01_r02g.erfh5" /*"/srv/logs/VW/BOOST_fill_sol_V01_OPT01_r02g_VV1.erfh5"*/)));

	QString format_path("/srv/logs/VW/erf_all.csv.format");
	PVRush::PVFormat f("", format_path);
	formats["custom"] = std::move(f);
	format = format_path;

	return true;
}

QString PVRush::PVInputTypeERF::name() const
{
	return QString("erf");
}

QString PVRush::PVInputTypeERF::human_name() const
{
	return QString("ERF import plugin");
}

QString PVRush::PVInputTypeERF::human_name_serialize() const
{
	return QString("ERF");
}

QString PVRush::PVInputTypeERF::internal_name() const
{
	return QString("07-elasticsearch");
}

QString PVRush::PVInputTypeERF::menu_input_name() const
{
	return QString("ERF...");
}

QString PVRush::PVInputTypeERF::tab_name_of_inputs(list_inputs const& in) const
{
	PVInputDescription_p query = in[0];
	return query->human_name();
}

bool PVRush::PVInputTypeERF::get_custom_formats(PVInputDescription_p /*in*/,
                                                hash_formats& /*formats*/) const
{
	return false;
}

QKeySequence PVRush::PVInputTypeERF::menu_shortcut() const
{
	return QKeySequence();
}
