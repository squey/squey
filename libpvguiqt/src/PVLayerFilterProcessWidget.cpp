/**
 * @file
 *
 * @copyright (C) Picviz Labs 2010-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <pvguiqt/PVLayerFilterProcessWidget.h>

#include <QDialog>
#include <QLabel>
#include <QPushButton>
#include <QMessageBox>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <inendi/PVStateMachine.h>
#include <inendi/widgets/PVArgumentListWidgetFactory.h>
#include <inendi/PVView.h>

PVGuiQt::PVLayerFilterProcessWidget::PVLayerFilterProcessWidget(Inendi::PVView* view,
                                                                PVCore::PVArgumentList& args,
                                                                Inendi::PVLayerFilter_p filter_p,
                                                                QWidget* parent)
    : QDialog(parent)
    , _view(view)
    , _filter_p(filter_p)
    , _presets_widget(nullptr)
    , _splitter(nullptr)
    , _help_btn(nullptr)
    , _args_org(args)
    , _has_apply(false)
{
	_args_widget = new PVWidgets::PVArgumentListWidget(
	    PVWidgets::PVArgumentListWidgetFactory::create_layer_widget_factory(*view), args, this);
	setWindowTitle("Filter properties...");
	setObjectName("PVLayerFilterProcessWidget");

	// Presets widget
	if (_filter_p->get_presets().can_have_presets()) {
		_presets_widget = new PVWidgets::PVPresetsWidget(tr("Presets"));
		_presets_widget->add_presets(_filter_p->get_presets().list_presets());
		connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_load_clicked_Signal, this,
		        &PVLayerFilterProcessWidget::load_preset_Slot);
		connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_new_clicked_Signal, this,
		        &PVLayerFilterProcessWidget::add_preset_Slot);
		connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_save_clicked_Signal, this,
		        &PVLayerFilterProcessWidget::save_preset_Slot);
		connect(_presets_widget, &PVWidgets::PVPresetsWidget::btn_remove_clicked_Signal, this,
		        &PVLayerFilterProcessWidget::remove_preset_Slot);
		connect(_presets_widget, &PVWidgets::PVPresetsWidget::preset_renamed_Signal, this,
		        &PVLayerFilterProcessWidget::rename_preset_Slot);
		_presets_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
		_presets_widget->setMinimumSize(QSize(0, 130));
	}

	// Args widget
	QVBoxLayout* args_widget_box_layout = new QVBoxLayout();
	QGroupBox* args_widget_box = new QGroupBox(tr("Filter"));
	args_widget_box_layout->addWidget(_args_widget);
	args_widget_box->setLayout(args_widget_box_layout);
	_args_widget->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);

	// Buttons
	_btn_layout = new QHBoxLayout();
	create_btns();
	set_btns_layout();
	connect_btns();

	// Splitter
	QVBoxLayout* main_layout = new QVBoxLayout();
	if (_filter_p->get_presets().can_have_presets()) {
		_splitter = new QSplitter(Qt::Vertical);
		_splitter->setChildrenCollapsible(false);
		_splitter->addWidget(args_widget_box);
		_splitter->addWidget(_presets_widget);
		_splitter->setStretchFactor(0, 4);
		_splitter->setStretchFactor(1, 1);
		main_layout->addWidget(_splitter);
	} else {
		main_layout->addWidget(args_widget_box);
	}

	qRegisterMetaType<Inendi::PVLayerFilter_p>("Inendi::PVLayerFilter_p");
	connect(this, &PVLayerFilterProcessWidget::layer_filter_error, this,
	        &PVLayerFilterProcessWidget::show_layer_filter_error);

	main_layout->addLayout(_btn_layout);
	setLayout(main_layout);
}

PVGuiQt::PVLayerFilterProcessWidget::~PVLayerFilterProcessWidget()
{
	_args_widget->deleteLater();
}

void PVGuiQt::PVLayerFilterProcessWidget::change_args(PVCore::PVArgumentList const& args)
{
	_args_widget->set_args_values(args);
}

void PVGuiQt::PVLayerFilterProcessWidget::add_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().add_preset(preset);
}

void PVGuiQt::PVLayerFilterProcessWidget::load_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().load_preset(preset);
	change_args(_filter_p->get_args());
}

void PVGuiQt::PVLayerFilterProcessWidget::remove_preset_Slot(const QString& preset)
{
	_filter_p->get_presets().del_preset(preset);
}

void PVGuiQt::PVLayerFilterProcessWidget::save_preset_Slot(const QString& preset)
{
	_filter_p->set_args(*_args_widget->get_args());
	_filter_p->get_presets().modify_preset(preset);
}

void PVGuiQt::PVLayerFilterProcessWidget::rename_preset_Slot(const QString& old_preset,
                                                             const QString& new_preset)
{
	_filter_p->get_presets().rename_preset(old_preset, new_preset);
}

void PVGuiQt::PVLayerFilterProcessWidget::create_btns()
{
	_apply_btn = new QPushButton(QIcon(":/green-check"), "Apply");
	_preview_btn = new QPushButton(QIcon(":/filter"), "Preview");
	_preview_btn->setDefault(true);
	_cancel_btn = new QPushButton(QIcon(":/red-cross"), "Cancel");
	_reset_btn = new QPushButton(QIcon(":/document-new"), "Reset");
	_reset_btn->setVisible(_filter_p->get_presets().can_have_presets());

	QString filter_desc = _filter_p->detailed_description();
	if (!filter_desc.isEmpty()) {
		_help_btn = new QPushButton(QIcon(":/help"), "Help");
	}
}

void PVGuiQt::PVLayerFilterProcessWidget::set_btns_layout()
{
	if (_help_btn) {
		_btn_layout->addWidget(_help_btn);
	}

	_btn_layout->addWidget(_cancel_btn);
	_btn_layout->addWidget(_reset_btn);
	_btn_layout->addWidget(_preview_btn);
	_btn_layout->addWidget(_apply_btn);
}

void PVGuiQt::PVLayerFilterProcessWidget::connect_btns()
{
	connect(_cancel_btn, SIGNAL(pressed()), this, SLOT(reject()));
	connect(_reset_btn, &QAbstractButton::pressed, this, &PVLayerFilterProcessWidget::reset_Slot);
	connect(_preview_btn, &QAbstractButton::pressed, this,
	        &PVLayerFilterProcessWidget::preview_Slot);
	connect(_apply_btn, &QAbstractButton::pressed, this, &PVLayerFilterProcessWidget::save_Slot);
	if (_help_btn) {
		QMessageBox* msgBox =
		    new QMessageBox(QMessageBox::Information, "Filter help",
		                    _filter_p->detailed_description(), QMessageBox::Ok, this);
		connect(_help_btn, &QAbstractButton::pressed, msgBox, &QDialog::exec);
	}
}

void PVGuiQt::PVLayerFilterProcessWidget::reject()
{
	if (!_has_apply) {
		QDialog::reject();
		return;
	}

	// Restore original arguments of this layer filter
	*_args_widget->get_args() = _args_org;

	// Restore view' post filter layer state
	_view->process_post_filter_layer();

	QDialog::reject();
}

void PVGuiQt::PVLayerFilterProcessWidget::save_Slot()
{
	// Force the current parameter widget to lose its focus (in case it has not
	// been updated yet !)
	_apply_btn->setFocus(Qt::MouseFocusReason);

	if (not _has_apply or _args_widget->args_changed()) {
		// Nothing already computed, do it now
		if (!process()) {
			// It has been canceled, so don't close the window !
			return;
		}
	}

	// FIXME : This is a Hack to commit colors in layer but not the selection.
	Inendi::PVLayer& current_selected_layer = _view->get_current_layer();
	_view->get_post_filter_layer().get_lines_properties().A2B_copy_restricted_by_selection(
	    current_selected_layer.get_lines_properties(),
	    _view->get_post_filter_layer().get_selection());

	_view->set_selection_view(_view->get_real_output_selection(), true);

	// Save last used filter
	_view->set_last_used_filter(_filter_p->registered_name());

	accept();
}

bool PVGuiQt::PVLayerFilterProcessWidget::process()
{
	_args_widget->force_submit();

	Inendi::PVLayerFilter_p filter_p = _filter_p->clone<Inendi::PVLayerFilter>();
	filter_p->set_args(*_args_widget->get_args());
	filter_p->set_view(_view);

	QWidget* parent_widget = isVisible() ? this : parentWidget();

	auto res = PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& /*pbox*/) {
		    try {
			    process_layer_filter(filter_p.get(), &_view->get_output_layer(),
			                         &_view->get_post_filter_layer());
		    } catch (const Inendi::PVLayerFilter::error& e) {

			    std::unique_lock<std::mutex> lk(_blocking_msg);
			    Q_EMIT layer_filter_error(
			        filter_p); // need to go back to GUI thread to show error widget
			    _cv.wait(lk);
		    }
		},
	    tr("Previewing filter..."), parent_widget);

	if (res != PVCore::PVProgressBox::CancelState::CONTINUE) {
		return false;
	}

	_has_apply = true;
	_args_widget->clear_args_state();

	return true;
}

void PVGuiQt::PVLayerFilterProcessWidget::preview_Slot()
{
	// Force the current parameter widget to lose its focus (in case it has not
	// been updated yet !)
	_preview_btn->setFocus(Qt::MouseFocusReason);

	/* Make sure the view is in its initial state but no need to notify views that
	 * something has changed because it will be done after the process() call
	 */
	_view->process_layer_stack(false);

	process();

	/* As process() updates view' post_filter_layer, we can not call
	 * PVView::process_post_filter_layer()
	 */
	_view->_update_output_selection.emit();
	_view->process_output_layer();
}

void PVGuiQt::PVLayerFilterProcessWidget::reset_Slot()
{
	change_args(_filter_p->get_default_args_for_view(*_view));
}

void PVGuiQt::PVLayerFilterProcessWidget::process_layer_filter(Inendi::PVLayerFilter* filter,
                                                               Inendi::PVLayer const* in_layer,
                                                               Inendi::PVLayer* out_layer)
{
	filter->set_output(out_layer);
	filter->operator()(*in_layer);
}

void PVGuiQt::PVLayerFilterProcessWidget::show_layer_filter_error(
    const Inendi::PVLayerFilter_p& filter)
{
	filter->show_error(this);
	_cv.notify_one();
}
