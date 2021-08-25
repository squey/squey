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

#include <inendi/PVPlotted.h>
#include <inendi/widgets/PVMappingPlottingEditDialog.h>
#include <pvguiqt/PVQPlotted.h>
#include <pvkernel/core/PVProgressBox.h>

bool PVGuiQt::PVQPlotted::edit_plotted(Inendi::PVPlotted& plotted, QWidget* parent)
{
	PVWidgets::PVMappingPlottingEditDialog* dlg =
	    new PVWidgets::PVMappingPlottingEditDialog(nullptr, &plotted, parent);
	if (dlg->exec() != QDialog::Accepted) {
		return false;
	}

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) { plotted.update_plotting(); },
	    QObject::tr("Updating plotting(s)..."), nullptr);

	return true;
}
