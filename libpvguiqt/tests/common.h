/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_TESTS_COMMON_H
#define PVGUIQT_TESTS_COMMON_H

#include <inendi/PVRoot_types.h>
#include <inendi/PVScene_types.h>
#include <inendi/PVSource_types.h>
#include <inendi/PVView_types.h>
#include <QString>

Inendi::PVSource_sp
get_src_from_file(Inendi::PVScene_sp scene, QString const& file, QString const& format);
Inendi::PVSource_sp
get_src_from_file(Inendi::PVRoot_sp root, QString const& file, QString const& format);
void init_random_colors(Inendi::PVView& view);

#endif
