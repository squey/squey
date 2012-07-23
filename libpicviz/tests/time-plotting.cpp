/**
 * \file time-plotting.cpp
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

#define FORMAT "pcre:test.time"
#define FILE "logs/test-time.log"

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

	retval = picviz_plotted_get_column_count(plotted);
	if (retval != 2) {
	  PVCore::log(PVCore::loglevel::critical, "Error: picviz_plotted_get_column_count: %d while it must be 2\n", retval);
	  return 1;
	}

	value = picviz_plotted_get_value(plotted, 0, 0);
	if (fabs(value - 0.612646) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.612646);
	  return 1;
	}
	value = picviz_plotted_get_value(plotted, 0, 1);
	if (fabs(value - 0.164145) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.164145);
	  return 1;
	}
	value = picviz_plotted_get_value(plotted, 1, 0);
	if (fabs(value - 0.612658) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.612658);
	  return 1;
	}
	value = picviz_plotted_get_value(plotted, 1, 1);
	if (fabs(value - 0.164145) > 0.00001) {
	  PVCore::log(PVCore::loglevel::critical, "error: value(0,0):%f while it must be %f\n", value, 0.164145);
	  return 1;
	}

	picviz_terminate();

	return 0;
}
