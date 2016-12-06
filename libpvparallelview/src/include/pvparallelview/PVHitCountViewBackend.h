/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2016
 */

#ifndef PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H
#define PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H

#include <inendi/PVPlottedNrawCache.h>

#include <pvparallelview/PVHitGraphBlocksManager.h>

namespace Inendi
{
class PVView;
class PVPlottedNrawCache;
} // namespace Inendi

namespace PVParallelView
{

class PVHitCountViewBackend
{
  public:
	PVHitCountViewBackend(const Inendi::PVView& view, const PVCol axis_index);

	Inendi::PVPlottedNrawCache& get_y_labels_cache() { return _y_labels_cache; }
	PVHitGraphBlocksManager& get_hit_graph_manager() { return _hit_graph_manager; }

  private:
	Inendi::PVPlottedNrawCache _y_labels_cache;
	PVHitGraphBlocksManager _hit_graph_manager;
};

} // namespace PVParallelView

#endif // PVPARALLELVIEW_PVHITCOUNTVIEWBACKEND_H
