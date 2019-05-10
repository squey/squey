/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#include <inendi/PVView.h>

#include <pvparallelview/PVHitCountViewBackend.h>

/*****************************************************************************
 * PVParallelView::PVHitCountViewBackend::PVHitCountViewBackend
 *****************************************************************************/

PVParallelView::PVHitCountViewBackend::PVHitCountViewBackend(const Inendi::PVView& view,
                                                             const PVCol axis)
    : _y_labels_cache(view, axis, 100)
    , _hit_graph_manager(view.get_parent<Inendi::PVPlotted>().get_column_pointer(axis),
                         view.get_row_count(),
                         2,
                         view.get_layer_stack_output_layer().get_selection(),
                         view.get_real_output_selection())
{
}
