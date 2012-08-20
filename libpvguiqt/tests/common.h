#ifndef PVGUIQT_TESTS_COMMON_H
#define PVGUIQT_TESTS_COMMON_H

#include <picviz/PVRoot_types.h>
#include <picviz/PVScene_types.h>
#include <picviz/PVSource_types.h>
#include <picviz/PVView_types.h>
#include <QString>

Picviz::PVSource_sp get_src_from_file(Picviz::PVScene_sp scene, QString const& file, QString const& format);
Picviz::PVSource_sp get_src_from_file(Picviz::PVRoot_sp root, QString const& file, QString const& format);

#endif
