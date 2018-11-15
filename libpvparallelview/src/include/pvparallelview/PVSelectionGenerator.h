/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVSELECTIONGENERATOR_H_
#define PVSELECTIONGENERATOR_H_

#include <pvparallelview/common.h>
#include <pvparallelview/PVAxisGraphicsItem.h>
#include <pvparallelview/PVLinesView.h>

namespace Inendi
{
class PVSelection;
class PVView;
} // namespace Inendi

namespace PVParallelView
{
class PVZonesManager;
class PVZoneTree;
class PVHitGraphBlocksManager;

namespace PVSelectionGenerator
{
constexpr unsigned int OR_MODIFIER = Qt::ShiftModifier;
constexpr unsigned int NAND_MODIFIER = Qt::ControlModifier;
constexpr unsigned int AND_MODIFIER = (Qt::ShiftModifier | Qt::ControlModifier);

void compute_selection_from_parallel_view_rect(int32_t width,
                                               PVZoneTree const& ztree,
                                               QRect rect,
                                               Inendi::PVSelection& sel);

uint32_t compute_selection_from_parallel_view_sliders(
    PVLinesView& lines_view,
    size_t zone_index,
    const typename PVAxisGraphicsItem::selection_ranges_t& ranges,
    Inendi::PVSelection& sel);

uint32_t compute_selection_from_hit_count_view_rect(const PVHitGraphBlocksManager& manager,
                                                    const QRectF& rect,
                                                    const uint32_t max_count,
                                                    Inendi::PVSelection& sel,
                                                    bool use_selectable);

uint32_t compute_selection_from_plotted_range(const uint32_t* plotted,
                                              PVRow nrows,
                                              uint64_t y_min,
                                              uint64_t y_max,
                                              Inendi::PVSelection& sel,
                                              Inendi::PVSelection const& layers_sel);

uint32_t compute_selection_from_plotteds_ranges(const uint32_t* y1_plotted,
                                                const uint32_t* y2_plotted,
                                                const PVRow nrows,
                                                const QRectF& rect,
                                                Inendi::PVSelection& sel,
                                                Inendi::PVSelection const& layers_sel);

void process_selection(Inendi::PVView& view,
                       Inendi::PVSelection const& sel,
                       bool use_modifiers = true);
} // namespace PVSelectionGenerator
;

namespace __impl
{
uint32_t compute_selection_from_plotted_range_sse(const uint32_t* plotted,
                                                  PVRow nrows,
                                                  uint64_t y_min,
                                                  uint64_t y_max,
                                                  Inendi::PVSelection& sel,
                                                  Inendi::PVSelection const& layers_sel);

uint32_t compute_selection_from_plotteds_ranges_sse(const uint32_t* y1_plotted,
                                                    const uint32_t* y2_plotted,
                                                    const PVRow nrows,
                                                    const QRectF& rect,
                                                    Inendi::PVSelection& sel,
                                                    Inendi::PVSelection const& layers_sel);

uint32_t
compute_selection_from_hit_count_view_rect_sse_invariant_omp(const PVHitGraphBlocksManager& manager,
                                                             const QRectF& rect,
                                                             const uint32_t max_count,
                                                             Inendi::PVSelection& sel,
                                                             bool use_selectable);
} // namespace __impl
} // namespace PVParallelView

#endif /* PVSELECTIONGENERATOR_H_ */
