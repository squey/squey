#include <pvhive/PVHive.h>
#include <pvhive/waxes/picviz/PVView.h>

PVHIVE_CALL_OBJECT_BLOCK_BEGIN()

IMPL_WAX(Picviz::PVView::process_from_selection, view, args)
{
	PVLOG_INFO("In WAX for Picviz::PVView::process_from_selection\n");
	call_object_default<Picviz::PVView, FUNC(Picviz::PVView::process_from_selection)>(view, args);
	refresh_observers(&view->get_output_layer());
}

PVHIVE_CALL_OBJECT_BLOCK_END()
