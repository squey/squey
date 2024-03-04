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

#include <QAction>
#include <QApplication>
#include <QPalette>
#include <QToolBar>
#include <QToolButton>
#include <QMenu>
#include <QComboBox>

#include <pvguiqt/PVWorkspace.h>
#include <pvguiqt/PVViewDisplay.h>
#include <pvguiqt/PVWorkspacesTabWidget.h>
#include <pvguiqt/PVProjectsTabWidget.h>
#include <pvguiqt/PVSimpleStringListModel.h>
#include <pvguiqt/PVStatusBar.h>
#include <pvguiqt/PVErrorsAndWarnings.h>

#include <squey/widgets/PVArgumentListWidgetFactory.h>
#include <squey/widgets/PVViewArgumentEditorCreator.h>

#include <pvkernel/core/PVAxisIndexType.h>
#include <pvkernel/core/PVZoneIndexType.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <pvkernel/widgets/PVArgumentListWidgetFactory.h>
#include <pvkernel/widgets/PVMouseButtonsLegend.h>

#include <pvkernel/core/qobject_helpers.h>

/******************************************************************************
 *
 * PVGuiQt::PVWorkspaceBase
 *
 *****************************************************************************/
uint64_t PVGuiQt::PVWorkspaceBase::_z_order_counter = 0;
bool PVGuiQt::PVWorkspaceBase::_drag_started = false;

PVGuiQt::PVWorkspaceBase::~PVWorkspaceBase() = default;

PVGuiQt::PVWorkspaceBase* PVGuiQt::PVWorkspaceBase::workspace_under_mouse()
{
	QList<PVWorkspaceBase*> active_workspaces;
	for (QWidget* top_widget : QApplication::topLevelWidgets()) {
		auto* w = qobject_cast<QMainWindow*>(top_widget);
		if (w) {
			for (PVProjectsTabWidget* project_tab_widget :
			     w->findChildren<PVProjectsTabWidget*>("PVProjectsTabWidget")) {
				PVSceneWorkspacesTabWidget* workspace_tab_widget =
				    project_tab_widget->current_workspace_tab_widget();
				if (workspace_tab_widget) {
					auto* workspace =
					    qobject_cast<PVWorkspaceBase*>(workspace_tab_widget->current_widget());
					if (workspace) {
						active_workspaces.append(workspace);
					}
				}
			}
		}
	}

	if (active_workspaces.size() == 0) {
		return nullptr;
	}

	PVWorkspaceBase* workspace = nullptr;
	int z_order = -1;

	for (PVWorkspaceBase* w : active_workspaces) {
		if (w->geometry().contains(w->mapFromGlobal(QCursor::pos()))) {
			if (w->z_order() > z_order) {
				z_order = w->z_order();
				workspace = w;
			}
		}
	}

	return workspace;
}

void PVGuiQt::PVWorkspaceBase::changeEvent(QEvent* event)
{
	QMainWindow::changeEvent(event);

	if (event->type() == QEvent::ActivationChange && isActiveWindow()) {
		_z_order_index = ++_z_order_counter;
	}
}

void PVGuiQt::PVWorkspaceBase::track_mouse_buttons_legend_changed(PVDisplays::PVDisplayIf& display_if, QWidget* widget)
{
	display_if._set_status_bar_mouse_legend.connect(
	[widget](QWidget* w, const PVWidgets::PVMouseButtonsLegend& legend){
		if (widget != w) return;
		QMainWindow* mw = PVCore::get_qobject_parent_of_type<QMainWindow*>(w);
		mw = PVCore::get_qobject_parent_of_type<QMainWindow*>(mw);
		PVGuiQt::PVStatusBar* status_bar = qobject_cast<PVGuiQt::PVStatusBar*>(mw->statusBar());

		status_bar->set_mouse_buttons_legend(legend);
	});

	display_if._clear_status_bar_mouse_legend.connect(
	[widget](QWidget* w){
		if (widget != w) return;
		QMainWindow* mw = PVCore::get_qobject_parent_of_type<QMainWindow*>(w);
		mw = PVCore::get_qobject_parent_of_type<QMainWindow*>(mw);
		PVGuiQt::PVStatusBar* status_bar = qobject_cast<PVGuiQt::PVStatusBar*>(mw->statusBar());

		status_bar->clear_mouse_buttons_legend();
	});
}

PVGuiQt::PVViewDisplay*
PVGuiQt::PVWorkspaceBase::add_view_display(Squey::PVView* view,
                                           QWidget* view_widget,
                                           PVDisplays::PVDisplayIf& display_if,
                                           bool delete_on_close /* = true*/,
                                           Qt::DockWidgetArea area /*= Qt::TopDockWidgetArea*/
)
{
	bool has_help_page = display_if.match_flags(PVDisplays::PVDisplayIf::HasHelpPage);
	bool can_be_central_display = display_if.match_flags(PVDisplays::PVDisplayIf::ShowInCentralDockWidget);
	auto* view_display =
	    new PVViewDisplay(view, view_widget, can_be_central_display, has_help_page, delete_on_close, this);

	// note : new connect syntax is causing a crash (Qt bug ?)
	connect(view_display, SIGNAL(destroyed(QObject*)), this, SLOT(display_destroyed(QObject*)));

	auto all_dock_widgets = findChildren<QDockWidget*>();
	auto other_in_same_area_it = std::ranges::find_if(all_dock_widgets, [this, area](auto* dw){ return dockWidgetArea(dw) == area; });
	if ((area & (Qt::LeftDockWidgetArea | Qt::RightDockWidgetArea)) && other_in_same_area_it != all_dock_widgets.end()) {
		tabifyDockWidget(*other_in_same_area_it, view_display);
		view_display->show();
		view_display->raise();
	} else {
		addDockWidget(area, view_display, Qt::Horizontal);
	}
	resizeDocks({view_display}, {500}, Qt::Horizontal); // Hack to fix children widgets sizes
	connect(view_display, &PVViewDisplay::try_automatic_tab_switch, this,
	        &PVWorkspaceBase::try_automatic_tab_switch);
	_displays.append(view_display);

	track_mouse_buttons_legend_changed(display_if, view_widget);

	return view_display;
}

PVGuiQt::PVViewDisplay* PVGuiQt::PVWorkspaceBase::set_central_display(Squey::PVView* view,
                                                                      QWidget* view_widget,
																	  bool has_help_page,
                                                                      bool delete_on_close)
{
	auto* view_display =
	    new PVViewDisplay(view, view_widget, true, has_help_page, delete_on_close, this);
	view_display->setStyleSheet("QDockWidget { font: bold }");
	view_display->setFeatures(QDockWidget::NoDockWidgetFeatures);
	view_display->setSizePolicy(
	    QSizePolicy(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding));
	setCentralWidget(view_display);

	_displays.append(view_display);

	return view_display;
}

void PVGuiQt::PVWorkspaceBase::switch_with_central_widget(
    PVViewDisplay* display_dock /* = nullptr */)
{
	if (!display_dock) {
		display_dock = (PVViewDisplay*)sender()->parent();
	}
	QWidget* display_widget = display_dock->widget();

	auto* central_dock = (PVViewDisplay*)centralWidget();

	if (central_dock) {
		QWidget* central_widget = central_dock->widget();

		// Exchange widgets
		central_dock->setWidget(display_widget);
		display_dock->setWidget(central_widget);

		// Exchange titles
		QString central_title = central_dock->windowTitle();
		central_dock->setWindowTitle(display_dock->windowTitle());
		display_dock->setWindowTitle(central_title);

		// Exchange bold
		central_dock->setStyleSheet("QDockWidget { font: bold }");
		display_dock->setStyleSheet("");

		// Exchange name functions
		central_dock->_name.swap(display_dock->_name);

		// Exchange views and view events registering
		Squey::PVView* central_view = central_dock->get_view();
		Squey::PVView* display_view = display_dock->get_view();
		central_dock->set_view(display_view);
		display_dock->set_view(central_view);

		// Exchange help button visibility
		bool central_help = central_dock->has_help_page();
		bool display_help = display_dock->has_help_page();
		central_dock->set_help_page_visible(display_help);
		display_dock->set_help_page_visible(central_help);
	} else {
		set_central_display(
			display_dock->get_view(),
			display_dock->widget(),
			central_dock->has_help_page(),
			display_dock->testAttribute(Qt::WA_DeleteOnClose)
		);
		removeDockWidget(display_dock);
	}
}

void PVGuiQt::PVWorkspaceBase::display_destroyed(QObject* object /*= 0*/)
{
	auto* display = (PVGuiQt::PVViewDisplay*)object;
	_displays.removeAll(display);
}

void PVGuiQt::PVWorkspaceBase::toggle_unique_source_widget(
    QAction* act, PVDisplays::PVDisplaySourceIf& display_if, Squey::PVSource* src)
{
	// All this should be the same than create_view_widget w/ a
	// PVCore::PVArgumentList passed to create_widget

	if (!src) {
		return;
	}

	QWidget* w = PVDisplays::get_widget(display_if, src);
	if (!w) {
		return;
	}

	PVViewDisplay* view_d = nullptr;
	for (PVViewDisplay* d : _displays) {
		if (d->widget() == w) {
			view_d = d;
			break;
		}
	}
	if (view_d) {
		view_d->setVisible(!view_d->isVisible());
	} else {
		view_d = add_view_display(
		    nullptr, w, display_if,
		    false);
		/* when the dock widget's "close" button is pressed, the
		 * associated QAction has to be unchecked
		 */
		connect(view_d, &QDockWidget::visibilityChanged, act, &QAction::setChecked);
	}
}

void PVGuiQt::PVWorkspaceBase::create_view_widget(PVDisplays::PVDisplayViewIf& display_if,
                                                  Squey::PVView* view,
                                                  std::vector<std::any> params)
{
	if (!view) {
		return;
	}

	QWidget* w = PVDisplays::get_widget(display_if, view, nullptr, params);
	auto area = display_if.default_position_hint();
	add_view_display(view, w,
	                 display_if, true,
	                 area ? area : Qt::TopDockWidgetArea);
}

/******************************************************************************
 *
 * PVGuiQt::PVSourceWorkspace
 *
 *****************************************************************************/
PVGuiQt::PVSourceWorkspace::PVSourceWorkspace(Squey::PVSource* source, QWidget* parent)
    : PVWorkspaceBase(parent), _source(source)
{
	// Invalid events widget
	if (source->get_invalid_evts().size() > 0) {
		auto* inv_elts_model =
		    new PVSimpleStringListModel(source->get_invalid_evts());
		_inv_evts_dlg = new PVGuiQt::PVListDisplayDlg(inv_elts_model, this);
		_inv_evts_dlg->setWindowTitle(tr("Invalid events"));
		_inv_evts_dlg->set_description(tr("There were invalid events during the extraction:"));
		_inv_evts_dlg->setAttribute(Qt::WA_DeleteOnClose, false);
	}

	_toolbar = new QToolBar(this);
	_toolbar->toggleViewAction()->setVisible(false);
	_toolbar->setFloatable(false);
	_toolbar->setMovable(false);
	_toolbar->setIconSize(QSize(24, 24));
	addToolBar(_toolbar);

	PVDisplays::visit_displays_by_if<PVDisplays::PVDisplaySourceIf>(
	    [this](PVDisplays::PVDisplaySourceIf& obj) {
		    auto* act = new QAction();
		    act->setCheckable(true);
		    act->setIcon(obj.toolbar_icon());
		    act->setToolTip(obj.tooltip_str());
		    _toolbar->addAction(act);

		    connect(act, &QAction::triggered,
		            [this, act, &obj] { toggle_unique_source_widget(act, obj, _source); });
	    },
	    PVDisplays::PVDisplayIf::ShowInToolbar & PVDisplays::PVDisplayIf::UniquePerParameters);

	class PVToolbarComboViews : public QComboBox
	{
		Squey::PVSource* _source;

	  public:
		PVToolbarComboViews(decltype(_source) source) : QComboBox(), _source(source)
		{
			fill_views();
			setToolTip("Current coherent viewset for this source");
			connect(this, &QComboBox::activated, [this](int index){
				if (index == count() - 1) {
					if (auto plotteds = _source->get_children<Squey::PVPlotted>(); plotteds.size() > 0) {
						// At time of writing, there is only one mapping/plotting per source
						plotteds.back()->emplace_add_child();
						fill_views();
						setCurrentIndex(count() - 2);
					}
				}
			});
		}
		void fill_views()
		{
			clear();
			QPixmap pm(24, 24);
			for (Squey::PVView* view : _source->get_children<Squey::PVView>()) {
				add_view_item(view, pm);
			}
			addItem(QIcon(":/more.png"), "Create new coherent viewset");
		}
		void add_view_item(Squey::PVView* view, QPixmap& pm)
		{
			pm.fill(view->get_color());
			addItem(QIcon(pm), QString::fromStdString(view->get_name()),
					QVariant::fromValue(view));
		}

	  protected:
		void showPopup() override
		{
			fill_views();
			QComboBox::showPopup();
		}
	};

	_toolbar_combo_views = new PVToolbarComboViews(_source);

	populate_display<PVDisplays::PVDisplayViewIf>();

	bool already_center = false;
	// Only one central widget is possible for QDockWidget.
	for (Squey::PVView* view : _source->get_children<Squey::PVView>()) {
		// Create default widgets
		PVDisplays::visit_displays_by_if<PVDisplays::PVDisplayViewIf>(
		    [&](PVDisplays::PVDisplayViewIf& obj) {
			    QWidget* w = PVDisplays::get_widget(obj, view);

			    const bool as_central = obj.default_position_as_central_hint();

			    const bool delete_on_close =
			        !obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters);
			    if (as_central && !already_center) {
				    already_center = true;
					bool has_help_page = obj.match_flags(PVDisplays::PVDisplayIf::HasHelpPage);
				    set_central_display(view, w, has_help_page, delete_on_close);
					track_mouse_buttons_legend_changed(obj, w);
			    } else {
				    Qt::DockWidgetArea pos = obj.default_position_hint();
				    if (as_central && already_center) {
					    pos = Qt::TopDockWidgetArea;
				    }
				    add_view_display(
				        view, w,
				        obj,
				        delete_on_close, pos);
			    }
		    },
		    PVDisplays::PVDisplayIf::DefaultPresenceInSourceWorkspace);
	}
}

bool PVGuiQt::PVSourceWorkspace::has_errors_or_warnings() const
{
	return get_source_invalid_evts_dlg() != nullptr or PVErrorsAndWarnings::invalid_columns_count(get_source()) > 0;
}

QString PVGuiQt::PVSourceWorkspace::source_type() const
{
	return _source->get_source_creator()->name();
}

const PVGuiQt::PVWorkspaceBase::PVViewWidgets&
PVGuiQt::PVWorkspaceBase::get_view_widgets(Squey::PVView* view)
{
	if (!_view_widgets.contains(view)) {
		PVViewWidgets widgets(view, this);
		return *(_view_widgets.insert(view, widgets));
	}
	return _view_widgets[view];
}

template <class T>
void PVGuiQt::PVSourceWorkspace::populate_display()
{
	PVDisplays::visit_displays_by_if<T>(
	    [&](T& obj) {
		    // if (!obj.match_flags(PVDisplays::PVDisplayIf::UniquePerParameters)) {
		    auto* btn = new QToolButton(_toolbar);
		    btn->setPopupMode(QToolButton::InstantPopup);
		    btn->setIcon(obj.toolbar_icon());
		    btn->setToolTip(obj.tooltip_str());
		    _toolbar->addWidget(btn);

		    connect(btn, &QToolButton::released, [this, &obj]() {
			    create_view_widget(obj,
			                       _toolbar_combo_views->currentData().value<Squey::PVView*>());
		    });
		    //}
	    },
	    PVDisplays::PVDisplayIf::ShowInToolbar);
}
