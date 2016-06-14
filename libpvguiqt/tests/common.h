/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_TESTS_COMMON_H
#define PVGUIQT_TESTS_COMMON_H

#include <inendi/PVScene_types.h>
#include <inendi/PVView_types.h>
#include <QString>

namespace Inendi
{
class PVRoot;
}

Inendi::PVSource&
get_src_from_file(Inendi::PVScene& scene, QString const& file, QString const& format);
Inendi::PVSource&
get_src_from_file(Inendi::PVRoot& root, QString const& file, QString const& format);

#endif
