/**
 * \file float-plotting.cpp
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#include <pvkernel/core/debug.h>
#include <pvkernel/rush/pvnormalizer.h>

#include <picviz/general.h>
#include <picviz/init.h>
#include <pvkernel/core/debug.h>

#include <picviz/datatreerootitem.h>
#include <picviz/scene.h>
#include <picviz/source.h>
#include <picviz/mapping.h>
#include <picviz/mapped.h>
#include <picviz/plotting.h>
#include <picviz/plotted.h>

#include <stdlib.h>
#include <math.h>

#define FORMAT "csv"
#define FILE "logs/collecte.csv"

int main(int argc, char **argv)
{
	picviz_datatreerootitem_t *dtri;
	picviz_scene_t *scene;
	picviz_source_t *source;
	picviz_mapping_t *mapping;
	picviz_mapped_t *mapped;
	picviz_plotting_t *plotting;
	picviz_plotted_t *plotted;

	int retval;
	float value;

#include "test-env.h"

	picviz_init(argc, NULL);

	dtri = picviz_datatreerootitem_new();
	scene = picviz_scene_new(dtri, const_cast<char*>("default"));
	source = picviz_source_new(scene);

	retval = picviz_source_file_append(source, (char *)FORMAT, (char *)FILE);

	mapping = picviz_mapping_new(source);
	mapped = picviz_mapped_make(mapping);
	plotting = picviz_plotting_new(mapped);
	plotted = picviz_plotted_build(plotting);

	picviz_plotted_debug(plotted);

	picviz_terminate();

	return 0;
}
