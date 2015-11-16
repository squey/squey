/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef FAKEPVVIEW_H_
#define FAKEPVVIEW_H_


#include <inendi/PVSelection.h>
#include <pvkernel/core/PVSharedPointer.h>

namespace Inendi
{

struct FakePVView
{
	typedef PVCore::PVSharedPtr<FakePVView> shared_pointer;

	void process_selection() {} ;

	PVSelection& get_view_selection() { return _sel; }
	PVSelection const& get_view_selection() const { return _sel; }

	PVSelection _sel;
};

typedef PVCore::PVSharedPtr<FakePVView> FakePVView_p;

}

#endif /* FAKEPVVIEW_H_ */
