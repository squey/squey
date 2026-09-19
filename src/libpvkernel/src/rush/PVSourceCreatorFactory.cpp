//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <qlist.h>
#include <memory>

#include "pvkernel/core/PVLogger.h"
#include "pvkernel/core/PVOrderedMap.h"
#include "pvkernel/rush/PVInputType.h"
#include "pvkernel/rush/PVSourceCreator.h"

PVRush::PVSourceCreator_p PVRush::PVSourceCreatorFactory::get_by_input_type(PVInputType_p in_t)
{
	QString itype = in_t->name();
	LIB_CLASS(PVRush::PVSourceCreator)& src_creators = LIB_CLASS(PVRush::PVSourceCreator)::get();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes const& list_creators = src_creators.get_list();
	LIB_CLASS(PVRush::PVSourceCreator)::list_classes::const_iterator itc;

	for (itc = list_creators.begin(); itc != list_creators.end(); itc++) {
		PVRush::PVSourceCreator_p sc = itc->value();
		if (sc->supported_type().compare(itype) != 0) {
			continue;
		}
		PVRush::PVSourceCreator_p sc_clone = sc->clone<PVRush::PVSourceCreator>();
		PVLOG_INFO("Found source for input type %s\n", qPrintable(in_t->human_name()));
		return sc_clone;
	}

	return {};
}
