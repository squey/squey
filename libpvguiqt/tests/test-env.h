/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <cstdlib>
#include <pvbase/general.h>
#include <inendi/common.h>

void init_env()
{
	setenv("PVFILTER_NORMALIZE_DIR", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins/normalize", 0);
	setenv("PVRUSH_NORMALIZE_HELPERS_DIR",
	       INENDI_SOURCE_DIRECTORY "/libpvkernel/plugins/normalize-helpers:./test-formats", 0);
	setenv("INENDI_CACHE_DIR", "./cache", 0);
	setenv("PVRUSH_INPUTTYPE_DIR", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins/input_types", 0);
	setenv("PVRUSH_SOURCE_DIR", INENDI_BUILD_DIRECTORY "/libpvkernel/plugins/sources", 0);
	setenv("INENDI_MAPPING_FILTERS_DIR",
	       INENDI_BUILD_DIRECTORY "/libinendi/plugins/mapping-filters", 0);
	setenv("INENDI_PLOTTING_FILTERS_DIR",
	       INENDI_BUILD_DIRECTORY "/libinendi/plugins/plotting-filters", 0);
	setenv("INENDI_LAYER_FILTERS_DIR", INENDI_BUILD_DIRECTORY "/libinendi/plugins/layer-filters",
	       0);
	Inendi::common::load_filters();
}
