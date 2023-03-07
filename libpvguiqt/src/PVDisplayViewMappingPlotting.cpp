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

#include <pvguiqt/PVDisplayViewMappingPlotting.h>

#include <squey/widgets/PVMappingPlottingEditDialog.h>
#include <squey/PVView.h>
#include <squey/PVPlotted.h>
#include <squey/PVMapped.h>

PVDisplays::PVDisplayViewMappingPlotting::PVDisplayViewMappingPlotting()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | UniquePerParameters,
                      "Mapping/Plotting",
                      QIcon(":/view-datatree"))
{
}

QWidget* PVDisplays::PVDisplayViewMappingPlotting::create_widget(Squey::PVView* view,
                                                                 QWidget* parent,
                                                                 Params const&) const
{
	auto* dlg = new PVWidgets::PVMappingPlottingEditDialog(&view->get_parent<Squey::PVMapped>(), &view->get_parent<Squey::PVPlotted>(), parent);

	dlg->connect(dlg, &QDialog::finished, [dlg] {
        delete dlg;
    });

	return dlg;
}
