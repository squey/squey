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

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVModdedIcon.h>

#include <squey/widgets/PVMappingPlottingEditDialog.h>
#include <squey/PVView.h>
#include <squey/PVPlotted.h>
#include <squey/PVMapped.h>

#include <QActionGroup>

PVDisplays::PVDisplayViewMappingPlotting::PVDisplayViewMappingPlotting()
    : PVDisplayViewIf(PVDisplayIf::ShowInToolbar | UniquePerParameters | ShowInCtxtMenu,
                      "Mapping/Plotting",
                      PVModdedIcon("mapping-scaling"))
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

	dlg->setWindowTitle(default_window_title(*view));
	return dlg;
}


void PVDisplays::PVDisplayViewMappingPlotting::add_to_axis_menu(
    QMenu& menu, PVCol axis, PVCombCol,
    Squey::PVView* view, PVDisplays::PVDisplaysContainer*)
{
    std::string axis_type = view->get_axes_combination().get_axis(axis).get_type().toStdString();

	Squey::PVMapped& mapped = view->get_parent<Squey::PVMapped>();
	Squey::PVMappingProperties& mpp =
	    mapped.get_properties_for_col(axis);

	menu.addSeparator();
	QMenu* chm = menu.addMenu("Change mapping to...");
	chm->setIcon(PVModdedIcon("mapping"));
	QActionGroup* chm_group = new QActionGroup(chm);

	for (auto& kvnode : LIB_CLASS(Squey::PVMappingFilter)::get().get_list()) {
		auto usable_list = kvnode.value()->list_usable_type();
		if (usable_list.empty() or usable_list.contains(axis_type)) {
			QAction* chm_filter = chm->addAction(kvnode.value()->get_human_name());
			chm_group->addAction(chm_filter);
			chm_filter->setCheckable(true);
            auto filter_name = kvnode.key().toStdString();
			chm_filter->setChecked(mpp.get_mode() == filter_name);
			QObject::connect(chm_filter, &QAction::triggered, [&mapped, &mpp, filter_name]() {
                mpp.set_mode(filter_name);
                PVCore::PVProgressBox::progress(
                    [&mapped](PVCore::PVProgressBox& /*pbox*/) { mapped.update_mapping(); },
                    QObject::tr("Updating mapping..."), nullptr);
            });
		}
	}

	Squey::PVPlotted& plotted = view->get_parent<Squey::PVPlotted>();
	Squey::PVPlottingProperties& plp =
	    plotted.get_properties_for_col(axis);

	QMenu* chp = menu.addMenu("Change plotting to...");
	chp->setIcon(PVModdedIcon("scaling"));
	menu.addSeparator();
	QActionGroup* chp_group = new QActionGroup(chp);

	for (auto& kvnode : LIB_CLASS(Squey::PVPlottingFilter)::get().get_list()) {
		auto usable_list = kvnode.value()->list_usable_type();
		if (usable_list.empty() or usable_list.contains(std::make_pair(axis_type, mpp.get_mode()))) {
			QAction* chp_filter = chp->addAction(kvnode.value()->get_human_name());
			chp_group->addAction(chp_filter);
			chp_filter->setCheckable(true);
            auto filter_name = kvnode.key().toStdString();
			chp_filter->setChecked(plp.get_mode() == filter_name);
			QObject::connect(chp_filter, &QAction::triggered,  [&plotted, &plp, filter_name]() {
                plp.set_mode(filter_name);
                PVCore::PVProgressBox::progress(
                    [&plotted](PVCore::PVProgressBox& /*pbox*/) { plotted.update_plotting(); },
                    QObject::tr("Updating plotting..."), nullptr);
            });
		}
	}
}
