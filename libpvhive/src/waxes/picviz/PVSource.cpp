#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVSource.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVSource::select_view, src, args)
{
	Picviz::PVView* old_cur_view = src->current_view();
	call_object_default<Picviz::PVSource, FUNC(Picviz::PVSource::select_view)>(src, args);
	refresh_observers(old_cur_view);
	refresh_observers(&std::get<0>(args));
}

PVHIVE_CALL_OBJECT_BLOCK_END()
