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

#include <pvkernel/rush/PVInputType.h>            // for PVInputType_p, PVInputType
#include <pvkernel/rush/PVSourceCreator.h>        // for PVSourceCreator_p, etc
#include <pvkernel/rush/PVSourceCreatorFactory.h> // for list_creators, etc
#include <pvkernel/rush/PVTests.h>
#include <pvkernel/core/PVClassLibrary.h> // for LIB_CLASS, etc
#include <iostream> // for operator<<, basic_ostream, etc

#include "pvkernel/rush/PVInputDescription.h"

namespace PVRush
{
class PVFormat;
} // namespace PVRush

bool PVRush::PVTests::get_file_sc(PVInputDescription_p /*file*/,
                                  PVRush::PVFormat const& /*format*/,
                                  PVSourceCreator_p& sc)
{
	// Load source plugins that take a file as input
	PVRush::PVInputType_p in_t = LIB_CLASS(PVRush::PVInputType)::get().get_class_by_name("file");
	if (!in_t) {
		std::cerr << "Unable to load the file input type plugin !" << std::endl;
		return false;
	}
	sc = PVRush::PVSourceCreatorFactory::get_by_input_type(in_t);

	return true;
}
