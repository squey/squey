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

#include <pvkernel/rush/PVInputDescription.h> // for PVInputDescription_p, etc
#include <pvkernel/rush/PVInputType.h>        // for PVInputType::list_inputs
#include <pvkernel/rush/PVSourceDescription.h>

#include <boost/iterator/indirect_iterator.hpp> // for indirect_iterator, etc
#include <boost/iterator/iterator_facade.hpp>   // for operator!=

#include <QFile>
#include <QList>
#include <QString>

#include <algorithm> // for equal
#include <memory>    // for __shared_ptr

bool PVRush::PVSourceDescription::operator==(const PVSourceDescription& other) const
{
	// FIXME: PVSourceCreator and PVFormat should have their own operator==

	return _inputs.size() == other._inputs.size() && //!\\ compare lists size before applying
	                                                 // std::equal to avoid segfault !
	       std::equal(boost::make_indirect_iterator(_inputs.begin()),
	                  boost::make_indirect_iterator(_inputs.end()),
	                  boost::make_indirect_iterator(other._inputs.begin())) &&
	       _source_creator_p->registered_name() == other._source_creator_p->registered_name() &&
	       _format.get_full_path() == other._format.get_full_path();
}
