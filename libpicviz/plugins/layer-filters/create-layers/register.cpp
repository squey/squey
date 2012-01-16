// Register the plugin in PVFilterLibrary
//

#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/rush/PVAxisTagsDec.h>

#include "PVLayerFilterCreateLayers.h"
#include "parse-config.h"

static int handle_create_layers_section(QString section_name, QString layer_name, QString regex_for_layer)
{
	QStringList section_tags = section_name.split(":");
	QString menu_name = QString("Create Layers/") + section_tags[0];

	REGISTER_CLASS_WITH_ARGS(menu_name, Picviz::PVLayerFilterCreateLayers, section_name);

        return 0;
}

int layers_get_config(QString filename)
{
        return create_layers_parse_config(filename, handle_create_layers_section);
}

// This method will be called by libpicviz
LibCPPExport void register_class()
{
	layers_get_config("create-layers.conf");
	// DECLARE_TAG(PVAXIS_TAG_DOMAIN, PVAXIS_TAG_DOMAIN_DESC, Picviz::PVLayerFilterCreateLayers);
}
