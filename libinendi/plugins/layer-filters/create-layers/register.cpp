/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

// Register the plugin in PVFilterLibrary
//

#include <pvbase/export.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include "PVLayerFilterCreateLayers.h"
#include "parse-config.h"

QStringList declared_tags;

static int append_tag(QString tag)
{
	for (int i = 0; i < declared_tags.size(); ++i) {
		if (!declared_tags.at(i).compare(tag)) {
			return 1;
		}
	}

	declared_tags << tag;
	DECLARE_TAG(tag, "", Inendi::PVLayerFilterCreateLayers);

	return 0;
}

static int handle_create_layers_section(QString section_name,
                                        QMap<QString, QStringList> layers_regex)
{
	QStringList section_tags = section_name.split(":");
	QString reg_name = QString("create-layers-") + section_tags[0];

	REGISTER_CLASS_WITH_ARGS(reg_name, Inendi::PVLayerFilterCreateLayers, section_name,
	                         layers_regex);
	if (section_tags.size() > 1) {
		append_tag(section_tags[1]);
	}

	return 0;
}

int layers_get_config(QString filename)
{
	return create_layers_parse_config(filename, handle_create_layers_section);
}

// This method will be called by libinendi
LibCPPExport void register_class()
{
	layers_get_config("create-layers.conf");
}