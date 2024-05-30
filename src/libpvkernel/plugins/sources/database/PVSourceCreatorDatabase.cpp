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

#include <pvkernel/rush/PVInput_types.h>

#include "../../common/database/PVDBQuery.h"

#include "PVSourceCreatorDatabase.h"
#include "PVDBSource.h"

PVRush::PVSourceCreatorDatabase::source_p
PVRush::PVSourceCreatorDatabase::create_source_from_input(PVInputDescription_p input) const
{
	auto* query = dynamic_cast<PVDBQuery*>(input.get());
	assert(query);
	source_p src = source_p(new PVRush::PVDBSource(*query, 100));

	return src;
}

QString PVRush::PVSourceCreatorDatabase::supported_type() const
{
	return {"database"};
}

QString PVRush::PVSourceCreatorDatabase::name() const
{
	return {"database"};
}
