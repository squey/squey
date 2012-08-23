/**
 * \file FakePVView.h
 *
 * Copyright (C) Picviz Labs 2010-2012
 */

#ifndef FAKEPVVIEW_H_
#define FAKEPVVIEW_H_


#include <picviz/PVSelection.h>
#include <pvkernel/core/PVSharedPointer.h>

namespace Picviz
{

struct FakePVView
{
	typedef PVCore::PVSharedPtr<FakePVView> shared_pointer;

	void process_selection() {} ;
	PVSelection& get_view_selection() { return _sel; }

	PVSelection _sel;
};

}

#endif /* FAKEPVVIEW_H_ */
