#ifndef PVPARALLELVIEW_TESTS_COMMON_H
#define PVPARALLELVIEW_TESTS_COMMON_H

#include <QString>
#include <picviz/PVView_types.h>

namespace PVParallelView {
class PVLibView;
}

PVParallelView::PVLibView* create_lib_view_from_args(int argc, char** argv);
int extra_param_start_at();
bool input_is_a_file();
void set_extra_param(int num, const char* usage_text);
void usage(const char* path);
Picviz::PVView_sp& get_view_sp();

#endif
