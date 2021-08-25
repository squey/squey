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

#include <pvkernel/core/PVClassLibrary.h>
#include <pvguiqt/common.h>

#include <pvguiqt/PVDisplaySourceDataTree.h>
#include <pvguiqt/PVDisplayViewAxesCombination.h>
#include <pvguiqt/PVDisplayViewListing.h>
#include <pvguiqt/PVDisplayViewLayerStack.h>
#include <pvguiqt/PVDisplayViewPythonConsole.h>

void PVGuiQt::common::register_displays()
{
	REGISTER_CLASS("guiqt_axes-combination", PVDisplays::PVDisplayViewAxesCombination);
	REGISTER_CLASS("guiqt_datatree", PVDisplays::PVDisplaySourceDataTree);
	REGISTER_CLASS("guiqt_layer-stack", PVDisplays::PVDisplayViewLayerStack);
	REGISTER_CLASS("guiqt_listing", PVDisplays::PVDisplayViewListing);
	REGISTER_CLASS("guiqt_pythonconsole", PVDisplays::PVDisplayViewPythonConsole);
}
