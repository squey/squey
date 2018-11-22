/**
 * @file
 *
 * @copyright (C) ESI Group INENDI 2018
 */

#include "PVSeriesViewSelectionRectangle.h"

namespace PVParallelView
{

PVSeriesViewSelectionRectangle::PVSeriesViewSelectionRectangle(QGraphicsScene* scene,
                                                               PVSeriesView& sv)
    : PVSelectionRectangle(scene), _sv(sv)
{
	// ctor
}

void PVSeriesViewSelectionRectangle::clear()
{
	PVSelectionRectangle::clear();
}

void PVSeriesViewSelectionRectangle::commit(bool use_selection_modifiers)
{
}

} // namespace PVParallelView
