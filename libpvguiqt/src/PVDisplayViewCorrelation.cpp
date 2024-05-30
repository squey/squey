/* * MIT License
 *
 * © Florent Chapelle, 2023
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include <pvkernel/widgets/PVModdedIcon.h>

#include <pvguiqt/PVDisplayViewCorrelation.h>

#include <squey/PVCorrelationEngine.h>
#include <squey/PVView.h>
#include <squey/PVRoot.h>
#include <squey/PVSource.h>
#include <squey/PVScaled.h>

#include <set>

auto PVDisplays::PVDisplayViewCorrelation::create_widget(Squey::PVView*, QWidget* parent, Params const&) const -> QWidget*
{
	assert("Should never get here");
	return new QWidget(parent);
}

void PVDisplays::PVDisplayViewCorrelation::add_to_axis_menu(
	QMenu& menu, PVCol axis, PVCombCol,
	Squey::PVView* view, PVDisplaysContainer*)
{

	const QString& this_axis_type = view->get_axes_combination().get_axis(axis).get_type();
	const QStringList& correlation_types_for_values = { "number_int8", "number_uint8", "number_int16", "number_uint16",
												        "number_int32", "number_uint32", "number_int64", "number_uint64",
												        "ipv4", "ipv6", "mac_address", "string" };
	const QStringList& correlation_types_for_range =  { "number_int8", "number_uint8", "number_int16", "number_uint16",
												        "number_int32", "number_uint32", "number_int64", "number_uint64",
												        "ipv4", "ipv6", "mac_address", /*"strings"*/
	                                                    "time", "duration", "number_float", "number_double" };

	// Don't show correlation menu for unsupported axes types
	if (not correlation_types_for_range.contains(this_axis_type) and
	    not correlation_types_for_values.contains(this_axis_type)) {
		return;
	}

	auto menu_add_correlation = new QMenu(QObject::tr("Bind parent axis with..."), &menu);
	menu_add_correlation->setAttribute(Qt::WA_TranslucentBackground);
	menu_add_correlation->setIcon(PVModdedIcon("link"));

	Squey::PVRoot& root = view->get_parent<Squey::PVRoot>();

	size_t total_compatible_views_count = 0;

	for (auto* source : root.get_children<Squey::PVSource>()) {

		size_t compatible_views_count = 0;

		// Don't allow correlation on same source
		if (source == root.current_source()) {
			continue;
		}

		QMenu* source_menu = new QMenu(QString::fromStdString(source->get_name()), &menu);

		size_t compatible_axes_count = 0;

		auto const views = source->get_children<Squey::PVView>();
		bool need_view_menu = views.size() > 1;
		for (Squey::PVView* child_view : views) {

			QMenu* view_menu = source_menu;

			// Don't create an intermediary view menu if there is only one view for this source
			if (need_view_menu) {
				view_menu = new QMenu(QString::fromStdString(child_view->get_name()), &menu);
				source_menu->addMenu(view_menu);
			}

			const Squey::PVAxesCombination& ac = child_view->get_axes_combination();
			std::set<PVCol> unique_comb_cols(ac.get_combination().begin(),
			                                 ac.get_combination().end());
			auto const& axes = child_view->get_parent<Squey::PVSource>().get_format().get_axes();
			for (PVCol original_col2 : unique_comb_cols) {
				const QString& axis_name = axes[original_col2].get_name();
				const QString& axis_type = axes[original_col2].get_type();

				// Don't show incompatible axes
				if (axis_type != this_axis_type) {
					continue;
				}

				QMenu* type_menu = new QMenu(axis_name, &menu);
				view_menu->addMenu(type_menu);

				// TODO : use QActionGroup for radio buttons

				auto add_correlation_f = [&](const QString& correlation_type_name, Squey::PVCorrelationType type){
					QAction* action = new QAction(correlation_type_name, &menu);
					action->setCheckable(true);

					Squey::PVCorrelation correlation{view, axis, child_view, original_col2, type};
					bool existing_correlation = root.correlations().exists(correlation);

					action->setChecked(existing_correlation);

					QObject::connect(action, &QAction::triggered, [=, &root]() {
						if (not existing_correlation) {
							root.correlations().add(correlation);
						} else {
							root.correlations().remove(correlation.view1);
						}
						// TODO refresh headers to show correlation icon right now
						// horizontalHeader()->viewport()->update();
					});

					type_menu->addAction(action);
				};

				if (correlation_types_for_values.contains(axis_type)) {
					add_correlation_f("distinct values", Squey::PVCorrelationType::VALUES);
				}
				if (correlation_types_for_range.contains(axis_type)) {
					add_correlation_f("minmax range", Squey::PVCorrelationType::RANGE);
				}

				compatible_axes_count++;
			}

			// Don't show view menu if there is no compatible axes
			if (compatible_axes_count > 0) {
				menu_add_correlation->addMenu(view_menu);
				compatible_views_count++;
			} else {
				delete view_menu;
			}
		}

		if (compatible_views_count == 0 && need_view_menu) {
			delete source_menu;
		}

		total_compatible_views_count += compatible_views_count;
	}

	// Don't show correlation menu if there is no compatible views
	if (total_compatible_views_count > 0) {
		menu.addMenu(menu_add_correlation);
	}
}
