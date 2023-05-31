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

#include <pvkernel/core/PVProgressBox.h>

#include <squey/PVView.h>

#include <pvparallelview/PVLibView.h>
#include <pvparallelview/PVParallelView.h>
#include <pvparallelview/PVFullParallelView.h>

#include <pvparallelview/PVDisplayViewFullParallel.h>

PVDisplays::PVDisplayViewFullParallel::PVDisplayViewFullParallel()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | PVDisplayIf::ShowInCentralDockWidget |
                          PVDisplayIf::DefaultPresenceInSourceWorkspace,
                      "Full parallel view",
                      QIcon(":/view-parallel-full"),
                      Qt::TopDockWidgetArea)
{
}

QWidget* PVDisplays::PVDisplayViewFullParallel::create_widget(Squey::PVView* view,
                                                              QWidget* parent,
                                                              Params const&) const
{
	PVParallelView::PVLibView* lib_view;

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    pbox.set_enable_cancel(false);
		    lib_view = PVParallelView::common::get_lib_view(*view);
		},
	    "Initializing full parallel view...", parent);

	return lib_view->create_view(parent);
	;
}
