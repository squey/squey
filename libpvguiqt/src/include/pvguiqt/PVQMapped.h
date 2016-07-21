/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVGUIQT_PVMAPPED_H
#define PVGUIQT_PVMAPPED_H

namespace Inendi
{
class PVMapped;
}

namespace PVGuiQt
{

struct PVQMapped {
	static bool edit_mapped(Inendi::PVMapped& plotted, QWidget* parent = nullptr);
};
}

#endif
