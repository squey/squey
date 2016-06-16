/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVHIVE_WAXES_INENDI_PVVIEW_H
#define PVHIVE_WAXES_INENDI_PVVIEW_H

#include <pvhive/PVWax.h>
#include <inendi/PVView.h>

// Processing waxes
//

DECLARE_WAX(Inendi::PVView::process_eventline)
DECLARE_WAX(Inendi::PVView::process_selection)
DECLARE_WAX(Inendi::PVView::process_visibility)
DECLARE_WAX(Inendi::PVView::process_from_selection)
DECLARE_WAX(Inendi::PVView::process_from_layer_stack)
DECLARE_WAX(Inendi::PVView::process_from_eventline)

#endif
