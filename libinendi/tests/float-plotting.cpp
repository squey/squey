/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvkernel/core/debug.h>
#include <pvkernel/rush/pvnormalizer.h>

#include <inendi/general.h>
#include <inendi/init.h>
#include <pvkernel/core/debug.h>

#include <inendi/datatreerootitem.h>
#include <inendi/scene.h>
#include <inendi/source.h>
#include <inendi/mapping.h>
#include <inendi/mapped.h>
#include <inendi/plotting.h>
#include <inendi/plotted.h>

#include <stdlib.h>
#include <math.h>

#define FORMAT "csv"
#define FILE "../../tests-files/inendi/collecte.csv"

int main(int argc, char **argv)
{
	inendi_datatreerootitem_t *dtri;
	inendi_scene_t *scene;
	inendi_source_t *source;
	inendi_mapping_t *mapping;
	inendi_mapped_t *mapped;
	inendi_plotting_t *plotting;
	inendi_plotted_t *plotted;

	int retval;
	float value;

#include "test-env.h"

	inendi_init(argc, NULL);

	dtri = inendi_datatreerootitem_new();
	scene = inendi_scene_new(dtri, const_cast<char*>("default"));
	source = inendi_source_new(scene);

	retval = inendi_source_file_append(source, (char *)FORMAT, (char *)FILE);

	mapping = inendi_mapping_new(source);
	mapped = inendi_mapped_make(mapping);
	plotting = inendi_plotting_new(mapped);
	plotted = inendi_plotted_build(plotting);

	inendi_plotted_debug(plotted);

	inendi_terminate();

	return 0;
}
