//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVPluginsLoad.h>

#include <pvkernel/filter/PVPluginsLoad.h>

#include <pvkernel/core/PVLogger.h>
#include <pvkernel/core/PVClassLibrary.h>

#include <squey/common.h>
#include <squey/plugins.h>

void Squey::common::load_filters()
{
	// PVRoot handle the filters
	load_layer_filters();
	load_mapping_filters();
	load_plotting_filters();

	// Load PVRush plugins
	PVRush::PVPluginsLoad::load_all_plugins();

	// Load PVFilter plugins
	PVFilter::PVPluginsLoad::load_all_plugins();
}

// Layer filters loading

/******************************************************************************
 *
 * Squey::common::load_layer_filters
 *
 *****************************************************************************/
int Squey::common::load_layer_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(squey_plugins_get_layer_filters_dir()), "layer_filter");
	if (ret == 0) {
		PVLOG_WARN("No layer filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d layer filters have been loaded.\n", ret);
	}

	return ret;
}

// Mapping filters loading

/******************************************************************************
 *
 * Squey::common::load_mapping_filters
 *
 *****************************************************************************/
int Squey::common::load_mapping_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(squey_plugins_get_mapping_filters_dir()), "mapping_filter");
	if (ret == 0) {
		PVLOG_WARN("No mapping filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d mapping filters have been loaded.\n", ret);
	}
	return ret;
}

// Plotting filters loading

/******************************************************************************
 *
 * Squey::common::load_plotting_filters
 *
 *****************************************************************************/
int Squey::common::load_plotting_filters()
{
	int ret = PVCore::PVClassLibraryLibLoader::load_class_from_dirs(
	    QString::fromStdString(squey_plugins_get_plotting_filters_dir()), "plotting_filter");
	if (ret == 0) {
		PVLOG_WARN("No plotting filters have been loaded !\n");
	} else {
		PVLOG_INFO("%d plotting filters have been loaded.\n", ret);
	}
	return ret;
}
