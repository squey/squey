/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef __INENDI_COMMON_H__
#define __INENDI_COMMON_H__

namespace Inendi
{
namespace common
{
void load_filters();

// Plugins loading
int load_layer_filters();
int load_mapping_filters();
int load_plotting_filters();
int load_axis_computation_filters();
int load_sorting_functions_filters();
} // namespace common
} // namespace Inendi

#endif
