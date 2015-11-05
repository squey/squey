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

#define FORMAT "pcre:test.time"
#define FILE "logs/test-time.log"

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

	retval = inendi_plotted_get_column_count(plotted);
	if (retval != 2) {
	  PVCore::log(PVCore::loglevel::critical, "Error: inendi_plotted_get_column_count: %d while it must be 2\n", retval);
	  return 1;
	}

	value = inendi_plotted_get_value(plotted, 0, 0);
	if (fabs(value - 0.612646) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.612646);
	  return 1;
	}
	value = inendi_plotted_get_value(plotted, 0, 1);
	if (fabs(value - 0.164145) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.164145);
	  return 1;
	}
	value = inendi_plotted_get_value(plotted, 1, 0);
	if (fabs(value - 0.612658) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.612658);
	  return 1;
	}
	value = inendi_plotted_get_value(plotted, 1, 1);
	if (fabs(value - 0.164145) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.164145);
	  return 1;
	}

	inendi_terminate();

	return 0;
}
