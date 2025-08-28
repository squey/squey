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

#include <QSplitter>
#include <QDesktopServices>
#include <QScreen>
#include <QInputDialog>
#include <QWidgetAction>

#include <PVFormatBuilderWidget.h>
#include <PVXmlTreeItemDelegate.h>
#include <PVXmlParamWidget.h>
#include <PVOptionsWidget.h>
#include <pvguiqt/PVInputTypeMenuEntries.h>

#include <pvkernel/rush/PVNormalizer.h>
#include <pvkernel/rush/PVNraw.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/rush/PVSourceCreatorFactory.h>
#include <pvkernel/rush/PVInput.h>
#include <pvkernel/filter/PVFieldSplitterChunkMatch.h>
#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVConverter.h>

#include <pvguiqt/PVAxesCombinationWidget.h>
#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvkernel/core/PVRecentItemsManager.h>


#include <boost/thread.hpp>
#include <memory>

QList<QUrl> App::PVFormatBuilderWidget::_original_shortcuts = QList<QUrl>();

#define FORMAT_BUILDER_TITLE (QObject::tr("Format builder"))
/******************************************************************************
 *
 * App::PVFormatBuilderWidget::PVFormatBuilderWidget
 *
 *****************************************************************************/
App::PVFormatBuilderWidget::PVFormatBuilderWidget(QWidget* parent)
    : QMainWindow(parent), _file_dialog(this)
{
	init(parent);
	setObjectName("PVFormatBuilderWidget");
	setAttribute(Qt::WA_DeleteOnClose, true);
}

void App::PVFormatBuilderWidget::closeEvent(QCloseEvent* event)
{
	if (myTreeView->getModel()->hasFormatChanged()) {
		QMessageBox msgBox(this);
		msgBox.setText("The format has been modified.");
		msgBox.setInformativeText("Do you want to save your changes?");
		msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Save);
		int ret = msgBox.exec();
		switch (ret) {
		case QMessageBox::Save:
			if (save()) {
				event->accept();
			} else {
				event->ignore();
			}
			break;
		case QMessageBox::Discard:
			event->accept();
			break;
		case QMessageBox::Cancel:
			event->ignore();
			break;
		default:
			// should never be reached
			break;
		}

	} else {
		event->accept();
	}
}

void App::PVFormatBuilderWidget::init(QWidget* /*parent*/)
{
	setWindowTitle(FORMAT_BUILDER_TITLE);

	auto main_splitter = new QSplitter(Qt::Vertical);
	/*
	 * ****************************************************************************
	 * Création of graphics elements.
	 * ****************************************************************************
	 */
	auto vb = new QVBoxLayout();
	vb->setContentsMargins(0, 0, 0, 0);
	auto vertical_splitter = new QSplitter(Qt::Horizontal);
	vbParam = new QVBoxLayout();
	vbParam->setSizeConstraint(QLayout::SetMinimumSize);

	// initialisation of the toolbar.
	actionAllocation();

	vb->addWidget(vertical_splitter);

	// the view
	myTreeView = new PVXmlTreeView(this);
	vertical_splitter->addWidget(myTreeView);

	// the model
	myTreeModel = new PVXmlDomModel(this);
	myTreeView->setModel(myTreeModel);

	auto vbParamWidget = new QWidget();
	vbParamWidget->setLayout(vbParam);
	vertical_splitter->addWidget(vbParamWidget);
	QSizePolicy sp(QSizePolicy::MinimumExpanding, QSizePolicy::MinimumExpanding);
	vertical_splitter->setSizePolicy(sp);
	vertical_splitter->setStretchFactor(vertical_splitter->indexOf(myTreeView), 1);
	vertical_splitter->setStretchFactor(vertical_splitter->indexOf(vbParamWidget), 40);
	// parameter board
	myParamBord_old_model = new PVXmlParamWidget(this);
	vbParam->addWidget(myParamBord_old_model);

	// param board plugin splitter
	myParamBord = &emptyParamBoard;
	vbParam->addWidget(myParamBord);

	// Create a table for the preview of the NRAW
	_nraw_model = new PVNrawListingModel();
	_nraw_widget = new PVNrawListingWidget(_nraw_model);
	_nraw_widget->connect_preview(this, &PVFormatBuilderWidget::slotExtractorPreview);
	connect(myTreeModel, &App::PVXmlDomModel::layoutChanged, this, &PVFormatBuilderWidget::slotExtractorPreview, Qt::QueuedConnection);
	_nraw_widget->connect_autodetect(this, [this](){slotAutoDetectAxesTypes(false);});
	_nraw_widget->connect_axes_name(this, &PVFormatBuilderWidget::set_axes_name_selected_row_Slot);
	_nraw_widget->connect_table_header(this,
	                                   &PVFormatBuilderWidget::slotItemClickedInMiniExtractor);

	// Put the vb layout into a widget and add it to the splitter
	_main_tab = new QTabWidget();
	auto vb_widget = new QWidget();
	vb_widget->setLayout(vb);
	_main_tab->addTab(vb_widget, tr("Format"));

	// Option tab
	_options_widget = new PVOptionsWidget();
	_main_tab->addTab(_options_widget, tr("Options"));

	_axes_comb_widget = new PVGuiQt::PVAxesCombinationWidget(myTreeModel->get_axes_combination());
	_main_tab->addTab(_axes_comb_widget, tr("Axes combination"));

	main_splitter->addWidget(_main_tab);

	// Tab widget for the NRAW
	auto nraw_tab = new QTabWidget();
	nraw_tab->addTab(_nraw_widget, tr("Format preview"));
	main_splitter->addWidget(nraw_tab);

	_inv_lines_widget = new QListWidget();
	nraw_tab->addTab(_inv_lines_widget, tr("Unmatched events"));

	auto main_layout = new QVBoxLayout();
	initToolBar(main_layout);
	main_layout->addWidget(main_splitter);

	auto central_widget = new QWidget();
	central_widget->setLayout(main_layout);

	setCentralWidget(central_widget);

	/* add of user/squey's path for formats
	 */
	setCentralWidget(central_widget);

	_file_dialog.setOption(QFileDialog::DontUseNativeDialog, true);
	_file_dialog.setNameFilters(QStringList{"Formats (*.format)", "All files (*.*)"});

	QList<QUrl> favorites = _file_dialog.sidebarUrls();

	/* As Qt keeps track of QFileDialog's state in an .ini file,
	 * to avoid polluting this state with P-I related paths, it
	 * is safe to save it before changing it.
	 */
	if (_original_shortcuts.length() == 0) {
		_original_shortcuts = favorites;
	}

	for (QString& s : PVRush::normalize_get_helpers_plugins_dirs(QString("text"))) {
		QFileInfo fi(QFileInfo(s).path());
		if (fi.isWritable()) {
			favorites.append(QUrl::fromLocalFile(s));
		}
	}

	_file_dialog.setSidebarUrls(favorites);

	// setWindowModality(Qt::ApplicationModal);

	/*
	 * ****************************************************************************
	 * Initialisation de toutes les connexions.
	 * ****************************************************************************
	 */
	lastSplitterPluginAdding = -1;
	initConnexions();

	slotUpdateToolsState();

	// AG: here, that's a bit tricky. We want our widget to have a maximize button,
	// but the only way to do that with Qt is to use setWindowFlag(Qt::Window).
	// According to Qt's documentation (and source code), as we are originally a widget,
	// this will set the pos of this window to absolute (0,0). That's not what we want,
	// because we want it centered, according to the main window's position (our parent).
	// So set the window flag and set the center os our gemotry to the center of our parent.
	// QRect geom = QRect(0,0,700,500);
	// setWindowFlags(Qt::Window);
	// geom.moveCenter(parent->geometry().center());
	// setGeometry(geom);
	QRect dr = QGuiApplication::screens()[0]->geometry();
	resize(dr.width() * 0.75, dr.height() * 0.75);
}
/******************************************************************************
 *
 * App::PVFormatBuilderWidget::~PVFormatBuilderWidget
 *
 *****************************************************************************/
App::PVFormatBuilderWidget::~PVFormatBuilderWidget()
{
	/* RH: restore the original shortcut list to prevent Qt from polluting
	 * its .ini file (under Unix: ~/.config/Trolltech.conf) by adding
	 * P-I formats related paths.
	 */
	_file_dialog.setSidebarUrls(_original_shortcuts);
	/*actionAddFilterAfter->deleteLater();
	actionAddRegExAfter->deleteLater();
	actionDelete->deleteLater();
	myTreeView->deleteLater();*/
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::actionAllocation
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::actionAllocation()
{
	actionAddAxisIn = new QAction("Add an axis", (QObject*)this);
	actionAddAxisIn->setIcon(PVModdedIcon("chart-simple"));
	actionAddFilterAfter = new QAction("Add a filter", (QObject*)this);
	actionAddFilterAfter->setIcon(PVModdedIcon("filter"));
	actionNameAxes = new QAction("Set axes name", (QObject*)this);
	actionNameAxes->setIcon(PVModdedIcon("pen-to-square"));
	actionAddRegExAfter = new QAction("add a RegEx", (QObject*)this);
	actionAddRegExAfter->setIcon(QIcon(":/add-regexp"));
	actionAddUrl = new QAction("add URL splitter", (QObject*)this);

	actionSave = new QAction("&Save format", (QObject*)this);
	actionSave->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_S));
	actionSave->setIcon(PVModdedIcon("floppy-disk"));
	actionSaveAs = new QAction("Save format as...", (QObject*)this);
	actionSaveAs->setShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S));
	actionSaveAs->setIcon(PVModdedIcon("floppy-disk-circle-arrow-right"));

	QMenu* import_menu = new QMenu();
	import_menu->setAttribute(Qt::WA_TranslucentBackground);
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(import_menu, this, SLOT(slotOpenLog()));
	buttonImport = new QToolButton();
    buttonImport->setToolTip("Import data");
    buttonImport->setPopupMode(QToolButton::InstantPopup);
	buttonImport->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_I));
	buttonImport->setIcon(PVModdedIcon("file-import"));
	buttonImport->setMenu(import_menu);

	QMenu* splitter_menu = new QMenu();
	splitter_menu->setAttribute(Qt::WA_TranslucentBackground);
	buttonSplitter = new QToolButton();
    buttonSplitter->setToolTip("Add a splitter");
    buttonSplitter->setPopupMode(QToolButton::InstantPopup);
	buttonSplitter->setIcon(PVModdedIcon("split"));
	buttonSplitter->setMenu(splitter_menu);
	for (const auto& it : LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_list()) {
		PVFilter::PVFieldsSplitterParamWidget_p pluginsSplitter = it.value();
		assert(pluginsSplitter);
		QAction* action = pluginsSplitter->get_action_menu(this);

		if (action) {
			action->setData(it.key());
			connect(action, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddSplitter);
			splitter_menu->addAction(action);
		}
	}

	QMenu* converter_menu = new QMenu();
	converter_menu->setAttribute(Qt::WA_TranslucentBackground);
	buttonConverter = new QToolButton();
    buttonConverter->setToolTip("Add a converter");
    buttonConverter->setPopupMode(QToolButton::InstantPopup);
	buttonConverter->setIcon(PVModdedIcon("swap"));
	buttonConverter->setMenu(converter_menu);
	for (const auto& it : LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_list()) {
		PVFilter::PVFieldsConverterParamWidget_p pluginsConverter = it.value();
		assert(pluginsConverter);
		QAction* action = pluginsConverter->get_action_menu(this);

		if (action) {
			action->setData(it.key());
			connect(action, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddConverter);
			converter_menu->addAction(action);
		}
	}

	actionDelete = new QAction("Delete", (QObject*)this);
	actionDelete->setIcon(PVModdedIcon("trash-xmark"));
	actionDelete->setShortcut(QKeySequence(Qt::Key_Delete));
	actionDelete->setEnabled(false);
	actionMoveDown = new QAction("Move down", (QObject*)this);
	actionMoveDown->setShortcut(QKeySequence(Qt::Key_Down));
	actionMoveDown->setIcon(PVModdedIcon("arrow-down-long"));
	actionMoveUp = new QAction("Move up", (QObject*)this);
	actionMoveUp->setShortcut(QKeySequence(Qt::Key_Up));
	actionMoveUp->setIcon(PVModdedIcon("arrow-up-long"));
	actionOpen = new QAction(tr("Open format..."), (QObject*)this);
	actionOpen->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_O));
	actionOpen->setIcon(PVModdedIcon("folder-open"));

	actionNewWindow = new QAction(tr("New window"), (QObject*)this);
	actionCloseWindow = new QAction(tr("Close window"), (QObject*)this);
	actionCloseWindow->setShortcut(QKeySequence(Qt::CTRL | Qt::Key_W));
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::initConnexions
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::initConnexions()
{
	// connexion to update the parameter board
	connect(myTreeView, &PVXmlTreeView::clicked, myParamBord_old_model, &PVXmlParamWidget::edit);
	// connexion to endable/desable items in toolsbar menu.
	connect(myTreeView, &PVXmlTreeView::clicked, this,
	        &PVFormatBuilderWidget::slotUpdateToolsState);

	// data has changed from tree
	connect(myTreeModel, &PVXmlDomModel::dataChanged, myTreeView,
	        &PVXmlTreeView::slotDataHasChanged);

	// When an item is clicked in the tree view, auto-select the good axis in the mini-extractor
	connect(myTreeView, &PVXmlTreeView::clicked, this,
	        &PVFormatBuilderWidget::slotItemClickedInView);

	/*
	 * Connexions for both menu and toolbar.
	 */
	connect(actionAddAxisIn, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddAxisIn);
	connect(actionAddFilterAfter, &QAction::triggered, this,
	        &PVFormatBuilderWidget::slotAddFilterAfter);
	connect(actionNameAxes, &QAction::triggered, this,
	        &PVFormatBuilderWidget::slotSetAxesName);
	connect(actionAddRegExAfter, &QAction::triggered, this,
	        &PVFormatBuilderWidget::slotAddRegExAfter);
	connect(actionDelete, &QAction::triggered, this, &PVFormatBuilderWidget::slotDelete);
	connect(actionMoveDown, &QAction::triggered, this, &PVFormatBuilderWidget::slotMoveDown);
	connect(actionMoveUp, &QAction::triggered, this, &PVFormatBuilderWidget::slotMoveUp);
	connect(actionNewWindow, &QAction::triggered, this, &PVFormatBuilderWidget::slotNewWindow);
	connect(actionCloseWindow, &QAction::triggered, this, &PVFormatBuilderWidget::close);
	connect(actionOpen, &QAction::triggered, this, &PVFormatBuilderWidget::slotOpen);
	connect(actionSave, &QAction::triggered, this, &PVFormatBuilderWidget::slotSave);
	connect(actionSaveAs, &QAction::triggered, this, &PVFormatBuilderWidget::slotSaveAs);

	connect(actionAddUrl, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddUrl);
	connect(myParamBord_old_model, &PVXmlParamWidget::signalNeedApply, this,
	        &PVFormatBuilderWidget::slotNeedApply);
	connect(myParamBord_old_model, &PVXmlParamWidget::signalSelectNext, myTreeView,
	        &PVXmlTreeView::slotSelectNext);
	connect(_options_widget, &PVOptionsWidget::first_line_changed, this,
	        [&](int first_line) { myTreeModel->set_first_line(first_line); });
	connect(_options_widget, &PVOptionsWidget::line_count_changed, this,
	        [&](int line_count) { myTreeModel->set_line_count(line_count); });
	connect(_options_widget, &PVOptionsWidget::python_script_updated, this,
	        [&](const QString& python_script, bool is_path, bool disabled)
	{
		myTreeModel->set_python_script(python_script, is_path, disabled);
	});

	// Connections for the axes combination editor
	connect(_main_tab, &QTabWidget::currentChanged, this,
	        &PVFormatBuilderWidget::slotMainTabChanged);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::initToolBar
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::initToolBar(QVBoxLayout* vb)
{

	auto tools = new QToolBar();

	tools->addAction(actionOpen);
	tools->addAction(actionSave);
	tools->addAction(actionSaveAs);
	tools->addWidget(buttonImport);
	tools->addSeparator();
	tools->addWidget(buttonSplitter);
	tools->addWidget(buttonConverter);
	tools->addAction(actionAddFilterAfter);
	tools->addAction(actionAddAxisIn);
	tools->addSeparator();
	tools->addAction(actionNameAxes);
	tools->addSeparator();
	tools->addAction(actionMoveDown);
	tools->addAction(actionMoveUp);
	tools->addSeparator();
	tools->addAction(actionDelete);


	vb->addWidget(tools);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddAxisIn
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddAxisIn()
{
	myTreeView->addAxisIn();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddFilterAfter
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddFilterAfter()
{
	myTreeView->addFilterAfter();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotSetAxesName
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotSetAxesName()
{
	bool ok;
	QString axes_name_text = QInputDialog::getText(this, tr("Set column names"),
	tr("Space separated column names:"), QLineEdit::Normal, "", &ok);
	if (ok) {
		axes_name_text.replace("\n", "");
		QStringList axes_name_list = axes_name_text.split(" ");
		myTreeModel->setAxesNames(axes_name_list);
	}
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddRegExAfter
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddRegExAfter()
{
	myTreeView->addRegExIn();
}
/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddSplitter
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddSplitter()
{
	auto* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	PVFilter::PVFieldsSplitterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(itype);
	PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	PVLOG_DEBUG("(App::PVFormatBuilderWidget::slotAddSplitter) type_name %s, %s\n",
	            qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
	myTreeView->addSplitter(in_t_cpy);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddConverter
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddConverter()
{
	auto* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	PVFilter::PVFieldsConverterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_class_by_name(itype);
	PVFilter::PVFieldsConverterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsConverterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	PVLOG_DEBUG("(App::PVFormatBuilderWidget::slotAddConverter) type_name %s, %s\n",
	            qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
	myTreeView->addConverter(in_t_cpy);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotAddUrl
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotAddUrl()
{
	myTreeView->addUrlIn();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotApplyModification
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotApplyModification()
{
	QModelIndex index;
	myTreeView->applyModification(myParamBord_old_model, index);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotDelete
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotDelete()
{
	if (!myTreeView->currentIndex().isValid()) {
		return;
	}

	QMessageBox msg(QMessageBox::Question, QString(), "Do you really want to delete it?",
	                QMessageBox::Yes | QMessageBox::No, this);

	if (msg.exec() == QMessageBox::Yes) {
		myTreeView->deleteSelection();
		myParamBord_old_model->drawForNo(QModelIndex());
		slotUpdateToolsState(myTreeView->currentIndex());
		Q_EMIT extractorPreview();
	}
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotMoveUp
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotMoveUp()
{
	myTreeView->moveUp();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotMoveDown
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotMoveDown()
{
	myTreeView->moveDown();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotNeedApply
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotNeedApply()
{
	QModelIndex index;
	myTreeView->applyModification(myParamBord_old_model, index);

	// Refresh params widget
	if (_log_input_type) {
		Q_EMIT extractorPreview();
	}
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotNewWindow
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotNewWindow()
{
	auto other = new PVFormatBuilderWidget;
	other->move(x() + 40, y() + 40);
	other->show();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotOpen
 *
 *****************************************************************************/
QString App::PVFormatBuilderWidget::slotOpen()
{
	_file_dialog.setWindowTitle("Load format from...");
	_file_dialog.setAcceptMode(QFileDialog::AcceptOpen);

	if (!_file_dialog.exec()) {
		return {};
	}

	const QString urlFile = _file_dialog.selectedFiles().at(0);

	if (urlFile.isEmpty() || (openFormat(urlFile) == false)) {
		return {};
	}

	return urlFile;
}

bool App::PVFormatBuilderWidget::save()
{
	// Take focus, so any currently edited argument will be set
	setFocus(Qt::MouseFocusReason);

	if (_cur_file.isEmpty()) {
		return saveAs();
	}

	if (!check_format_validity()) {
		return false;
	}

	bool save_xml = myTreeModel->saveXml(_cur_file);
	if (save_xml) {
		PVCore::PVRecentItemsManager::get().add<PVCore::Category::EDITED_FORMATS>(_cur_file);

		check_for_new_time_formats();

		return true;
	}

	QMessageBox err(QMessageBox::Question, tr("Error while saving format"),
	                tr("Unable to save the changes to %1. Do you want to save this format to "
	                   "another location ?")
	                    .arg(_cur_file),
	                QMessageBox::Yes | QMessageBox::No);
	if (err.exec() == QMessageBox::No) {
		return false;
	}

	return saveAs();
}

void App::PVFormatBuilderWidget::check_for_new_time_formats()
{
	try {
		const std::unordered_set<std::string> time_formats =
		    get_format_from_dom().get_time_formats();
		std::unordered_set<std::string> supported_time_formats =
		    PVRush::PVTypesDiscoveryOutput::supported_time_formats();
		std::unordered_set<std::string> new_time_formats;

		for (const std::string& time_format : time_formats) {
			auto it = supported_time_formats.find(time_format);
			if (it == supported_time_formats.end() &&
			    time_format.find("epoch") == std::string::npos) {
				new_time_formats.emplace(time_format);
			}
		}

		if (not new_time_formats.empty()) {
			QStringList time_formats_list;
			for (const std::string& time_format : new_time_formats) {
				time_formats_list << QString::fromStdString(time_format);
			}

			bool multi = new_time_formats.size() > 1;
			bool add_time_formats =
			    QMessageBox::question(
			        this, QString("New time format") + (multi ? "s" : "") + " detected",
			        QString("The following time format") + (multi ? "s are" : " is") +
			            " not currently enabled in axes type autodetection :" + "<br><br><i>" +
			            time_formats_list.join("<br>") + "<br><br></i>Do you want to enable " +
			            (multi ? "them" : "it") + " in future autodetections?",
			        QMessageBox::Yes | QMessageBox::No, QMessageBox::Yes) == QMessageBox::Yes;

			if (add_time_formats) {
				PVRush::PVTypesDiscoveryOutput::append_time_formats(new_time_formats);
			}
		}
	} catch (const PVRush::PVFormatInvalid& e) {
		// there is no time format to gather if the format is invalid
	}
}

bool App::PVFormatBuilderWidget::check_format_validity()
{
	if (myTreeModel->get_axes_count() >= 2) {
		return true;
	}

	auto res = QMessageBox::warning(this, "Invalid format...",
	                                "Your format has less than 2 axes and will "
	                                "not be usable to import any data.<br><br>"
	                                "Do you want to save it anyway?",
	                                QMessageBox::Yes | QMessageBox::No);

	return res == QMessageBox::Yes;
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotSave
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotSave()
{
	save();
}

bool App::PVFormatBuilderWidget::saveAs()
{
	setFocus(Qt::MouseFocusReason);

	if (!check_format_validity()) {
		return false;
	}

	QModelIndex index;
	myTreeView->applyModification(myParamBord_old_model, index);

	_file_dialog.setWindowTitle("Save format to...");
	_file_dialog.setAcceptMode(QFileDialog::AcceptSave);

	if (!_file_dialog.exec()) {
		return false;
	}

	check_for_new_time_formats();

	const QString urlFile = _file_dialog.selectedFiles().at(0);
	if (!urlFile.isEmpty()) {
		if (myTreeModel->saveXml(urlFile)) {
			_cur_file = urlFile;
			setWindowTitleForFile(urlFile);
			PVCore::PVRecentItemsManager::get().add<PVCore::Category::EDITED_FORMATS>(urlFile);
			return true;
		}
	}

	return false;
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotSaveAs
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotSaveAs()
{
	saveAs();
}

void App::PVFormatBuilderWidget::slotAutoDetectAxesTypes(bool handle_header /* = true */)
{
	static constexpr const size_t mega = 1024 * 1024;

	PVRow start, end;
	_nraw_widget->get_autodetect_args(start, end);

	bool is_row_count_known = end != PVROW_INVALID_VALUE;
	if (not is_row_count_known) {
		end = EXTRACTED_ROW_COUNT_LIMIT;
	}

	PVRush::PVTypesDiscoveryOutput type_discovery_output;

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    PVRush::PVFormat format = get_format_from_dom();
		    QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
		    list_inputs << _log_input;

		    PVRush::PVExtractor extractor(format, type_discovery_output, _log_sc, list_inputs);

		    pbox.set_cancel2_btn_text("Stop");
		    pbox.set_cancel_btn_text("Cancel");

		    if (is_row_count_known) {
			    pbox.set_maximum(end - start);
		    } else {
			    pbox.set_maximum(extractor.max_size() / mega);
		    }
		    pbox.set_enable_cancel(true);

		    PVRush::PVControllerJob_p job = extractor.process_from_agg_idxes(start, end);

		    try {
			    // update the status of the progress bar
			    while (job->running()) {
				    if (is_row_count_known) {
					    pbox.set_value(job->status());
				    } else {
					    pbox.set_value(job->get_value() / mega);
				    }
				    pbox.set_extended_status(QString("Processed rows: %L1").arg(job->status()));
				    boost::this_thread::interruption_point();
				    boost::this_thread::sleep(boost::posix_time::milliseconds(50));
			    }
		    } catch (boost::thread_interrupted) {
			    job->cancel();
		    }

		    job->wait_end();

		    if (pbox.get_cancel_state() == PVCore::PVProgressBox::CancelState::CANCEL) {
			    return;
		    }
	    },
	    QObject::tr("Autodetecting axes types..."), nullptr);

	// TODO : it would be nice to ignore axes set by user...

	// Update format with discovered axes types
	QDomElement& dom = myTreeModel->getRootDom();
	QDomNodeList axes = dom.elementsByTagName("axis");
	bool has_header = false;
	for (PVCol i(0); i < axes.size(); i++) {
		QDomElement ax = axes.at(i).toElement();
		std::string type;
		std::string type_format;
		std::string axe_name;
		std::tie(type, type_format, axe_name) = type_discovery_output.type_desc(i);
		ax.setAttribute("type", type.c_str());
		ax.setAttribute("type_format", type_format.c_str());
		if (not axe_name.empty() and myTreeModel->get_first_line() == 0) {
			ax.setAttribute("name", axe_name.c_str());
			has_header = true;
		}
	}

	if (handle_header) {
		if (has_header) {
			if (QMessageBox::question(
					this, "Header detected",
					"A header was detected: use it to fill column names?") ==
				QMessageBox::Yes) {
				_options_widget->set_lines_range(1, myTreeModel->get_line_count());
			}
		}
		else {
			if (QMessageBox::question(
					this, "No header was detected",
					"No header was detected: do you want to manually enter column names?") ==
				QMessageBox::Yes) {
				slotSetAxesName();
			}
		}
	}

	// Refresh params widget
	myTreeView->refresh();

	// Run preview
	slotExtractorPreview();
}

void App::PVFormatBuilderWidget::update_types_autodetection_count(
    const PVRush::PVFormat& format)
{
	static constexpr const size_t total_fields = 100000;
	static constexpr const size_t multiple_to_round = 100;

	PVCol column_count = (PVCol)format.get_axes().size();
	assert(column_count != 0);

	size_t row_count =
	    (((total_fields / column_count) + multiple_to_round - 1) / multiple_to_round) *
	    multiple_to_round;

	_nraw_widget->set_autodetect_count(row_count);
}

void App::PVFormatBuilderWidget::setWindowTitleForFile(QString const& path)
{
	// Change the window title with the filename of the format
	QFileInfo fi(path);
	setWindowTitle(FORMAT_BUILDER_TITLE + QString(" - ") + fi.fileName());
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotUpdateToolsState
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotUpdateToolsState(const QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* node = myTreeModel->nodeFromIndex(index);

	bool force_root_action = not index.parent().isValid();

	if (node->getDom().tagName() == "field") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(true);
		actionAddAxisIn->setEnabled(true);
		buttonSplitter->setEnabled(true);
		buttonConverter->setEnabled(true);
		actionDelete->setEnabled(false);
	} else if (node->getDom().tagName() == "axis") {
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(false);
		buttonConverter->setEnabled(false);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "filter" || node->getDom().tagName() == "converter") {
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(force_root_action);
		buttonConverter->setEnabled(force_root_action);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "splitter") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(false);
		buttonConverter->setEnabled(force_root_action);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "RegEx") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(false);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(false);
		buttonConverter->setEnabled(false);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "url") {
		actionAddFilterAfter->setEnabled(false);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(false);
		buttonConverter->setEnabled(false);
		actionDelete->setEnabled(true);
	} else {
		actionAddFilterAfter->setEnabled(true);
		actionAddAxisIn->setEnabled(false);
		buttonSplitter->setEnabled(true);
		buttonConverter->setEnabled(true);
		actionDelete->setEnabled(false);
	}
}

void App::PVFormatBuilderWidget::get_source_creator_from_inputs(
    const PVRush::PVInputDescription_p input,
    const PVRush::PVInputType_p& input_type,
    PVRush::PVSourceCreator_p& source_creator,
    PVRush::PVRawSourceBase_p& raw_source_base) const
{
	// Get the first input selected
	PVLOG_DEBUG("Input: %s\n", qPrintable(input_type->human_name_of_input(input)));

	// Get list of inputs from the plugin.
	PVRush::PVSourceCreator_p sc = PVRush::PVSourceCreatorFactory::get_by_input_type(input_type);
	try {
		source_creator = sc;
		// The moni-extractor use the discovery source, as not that much processing is
		// done (it can be handle locally for instance !)
		raw_source_base = source_creator->create_source_from_input(input);
	} catch (PVRush::PVFormatInvalid& e) {
		source_creator.reset();
	} catch (std::ios_base::failure const& e) {
		// File can't be found, looks for another type.
		source_creator.reset();
	}
	// If the log_source can't be create, look for another source.
	if (raw_source_base.get() == nullptr) {
		source_creator.reset();
	}
}

PVRush::PVFormat App::PVFormatBuilderWidget::load_log_and_guess_format(
    const PVRush::PVInputDescription_p input, const PVRush::PVInputType_p& input_type)
{
	_log_input_type = input_type;

	_inputs.clear();
	_inputs.push_back(input);

	load_log(FORMATBUILDER_EXTRACT_START_DEFAULT, FORMATBUILDER_EXTRACT_END_DEFAULT);

	_cur_file = input->human_name() + ".format";

	return get_format_from_dom();
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::guess_format
 *
 *****************************************************************************/
PVRush::PVFormat
App::PVFormatBuilderWidget::guess_format(const PVRush::PVRawSourceBase_p& raw_source_base,
                                                 PVXmlDomModel& tree_model) const
{
	// Guess first splitter and add it to the dom before parsing it !
	// The dom is the reference in here.

	PVLOG_DEBUG("(format_builder) trying to guess first splitter...");
	PVCol naxes;
	PVFilter::PVFieldsSplitter_p sp =
	    PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(raw_source_base, naxes);
	if (!sp) {
		// No splitter matches, just do nothing
		return {};
	}

	// Get the widget that comes with the splitter. TODO: do better than that
	QString type_name = sp->type_name();
	QString filter_name = sp->registered_name();
	PVFilter::PVFieldsSplitterParamWidget_p sp_widget =
	    LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(filter_name);
	if (!sp_widget) {
		PVLOG_WARN("Filter '%s' of type '%s' has no associated widget !\n", qPrintable(type_name),
		           qPrintable(filter_name));
		return {};
	}

	// Then we need to create 'naxes' children
	QStringList axes_name;
	for (PVCol i(0); i < naxes; i++) {
		axes_name << QString("Axis %1").arg(i + 1);
	}

	sp_widget->set_child_count(naxes);

	PVRush::PVXmlTreeNodeDom* node =
	    tree_model.addSplitterWithAxes(tree_model.index(0, 0, QModelIndex()), sp_widget, axes_name);
	node->setFromArgumentList(sp->get_args());

	// Set proper strings mapping
	for (PVRush::PVXmlTreeNodeDom* elt : node->getChildren()) {
		for (PVRush::PVXmlTreeNodeDom* axis : elt->getChildren()) {
			if (axis->attribute("type") == "string") {
				axis->setMappingProperties("string", {}, {});
			}
		}
	}

	return PVRush::PVFormat(tree_model.getRootDom());
}

PVRush::PVFormat
App::PVFormatBuilderWidget::guess_format(const PVRush::PVInputDescription_p input,
                                                 const PVRush::PVInputType_p& input_type) const
{
	PVRush::PVSourceCreator_p source_creator;
	PVRush::PVRawSourceBase_p raw_source_base;

	get_source_creator_from_inputs(input, input_type, source_creator, raw_source_base);

	PVXmlDomModel tree_model;

	return guess_format(raw_source_base, tree_model);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::load_log
 *
 *****************************************************************************/

void App::PVFormatBuilderWidget::load_log(PVRow rstart, PVRow rend)
{
	// If no files where selected, ask for one.
	if (_inputs.isEmpty()) {

		QString choosenFormat;
		PVRush::hash_formats formats;

		// This case is only encountered when a source is loaded from the menu
		PVCore::PVArgumentList args;
		if (!_log_input_type->create_widget(formats, _inputs, choosenFormat, args, this)) {
			return; // This means that the user pressed the "cancel" button
		}
		assert(not _inputs.empty() && "At least one file have to be selected");
	}

	bool has_error = false;

	try {
		// Get the first input selected
		_log_input = _inputs.front();
		PVLOG_DEBUG("Input: %s\n", qPrintable(_log_input_type->human_name_of_input(_log_input)));

		// Pre discover the input w/ the source creators
		_log_sc.reset();

		get_source_creator_from_inputs(_log_input, _log_input_type, _log_sc, _log_source);

		// No source found to load data. Show an error and quit.
		if (!_log_sc) {
			_log_input = PVRush::PVInputDescription_p();
			QMessageBox box(QMessageBox::Critical, tr("Error"),
			                tr("No input plugins can manage the source file '%1'. Aborting...")
			                    .arg(_log_input_type->human_name_of_input(_log_input)));
			box.show();
			return;
		}

		// First extraction
		PVRush::PVFormat format;
		if (is_dom_empty()) {
			format = guess_format(_log_source, *myTreeModel);

			PVCol naxes;
			PVFilter::PVFieldsSplitter_p sp =
			    PVFilter::PVFieldSplitterChunkMatch::get_match_on_input(_log_source, naxes);
			if (sp) {
				// Ok, we got a match, add it to the dom.
				QString first_input_name = _log_input_type->human_name_of_input(_inputs.front());
				PVLOG_INFO("(format_builder) For input '%s', found a splitter that creates %d "
				           "axes. Arguments:\n",
				           qPrintable(first_input_name), naxes);
				PVCore::dump_argument_list(sp->get_args());

				QString msg =
				    tr("It appears that the %1 splitter can process '%2' and create %3 fields.\n\n")
				        .arg(sp->registered_name())
				        .arg(first_input_name)
				        .arg(naxes);
				msg += tr("Do you want to automatically add that splitter to the format ?");
				QMessageBox ask_auto(QMessageBox::Question, tr("Filter automatically found"), msg,
				                     QMessageBox::Yes | QMessageBox::No, this);
				if (ask_auto.exec() == QMessageBox::No) {
					return;
				}
			}
			if (not is_dom_empty()) {
				update_types_autodetection_count(format);
				slotAutoDetectAxesTypes();
				format.set_first_line(myTreeModel->get_first_line());
			} else {
				QMessageBox::information(this, "Splitter not automatically detected",
				                         "The splitter was not automatically detected.\nYou have "
				                         "to manually define a splitter.");
				return;
			}
		} else {
			format = get_format_from_dom();
			update_types_autodetection_count(format);
		}

		_nraw = std::make_unique<PVRush::PVNraw>();
		_nraw_output = std::make_unique<PVRush::PVNrawOutput>(*_nraw);
		QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
		list_inputs << _log_input;

		_log_extract.reset(new PVRush::PVExtractor(format, *_nraw_output, _log_sc, list_inputs));

		update_table(rstart, rend);

	} catch (PVRush::PVInputException& e) {
		_nraw_widget->set_error_message("Error while importing a source: " + QString(e.what()));
		has_error = true;
	} catch (PVFilter::PVFieldsFilterInvalidArguments const& e) {
		_nraw_widget->set_error_message(e.what());
		has_error = true;
	} catch (PVRush::PVFormatInvalid const& e) {
		_nraw_widget->set_error_message("The current format is not valid. We can't perform an import : " + QString(e.what()));
		has_error = true;
	} catch (PVRush::PVFormatInvalidTime const& e) {
		_nraw_widget->set_error_message(e.what());
		has_error = true;
	} catch (PVRush::PVConverterCreationError const& e) {
		_nraw_widget->set_error_message(QString("Unsupported charset : ") + e.what());
		has_error = true;
	}

	if (has_error) {
		// make sure to reset possibly altered member variables.
		_log_input = PVRush::PVInputDescription_p();
		_log_source.reset();
		_log_sc.reset();
		_nraw = std::make_unique<PVRush::PVNraw>();
		_nraw_output = std::make_unique<PVRush::PVNrawOutput>(*_nraw);
		return;
	}
	_nraw_widget->unset_error_message();

	// Tell the NRAW widget that the input has changed
	_nraw_widget->set_last_input(_log_input_type, _log_input);
}

/******************************************************************************
 *
 * App::PVFormatBuilderWidget::slotOpenLog
 *
 *****************************************************************************/
void App::PVFormatBuilderWidget::slotOpenLog()
{
	_log_input_type = PVGuiQt::PVInputTypeMenuEntries::input_type_from_action((QAction*)sender());

	_inputs.clear();

	load_log(FORMATBUILDER_EXTRACT_START_DEFAULT, FORMATBUILDER_EXTRACT_END_DEFAULT);
}

PVRush::PVFormat App::PVFormatBuilderWidget::get_format_from_dom() const
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return PVRush::PVFormat{rootDom};
}

void App::PVFormatBuilderWidget::update_table(PVRow start, PVRow end)
{
	assert(_log_extract);
	assert(end > start);

	// Clear the filter previous data
	myTreeModel->clearFiltersData();

	// Update the data displaying of the filter param widgers
	myTreeModel->updateFiltersDataDisplay();

	PVRush::PVControllerJob_p job = _log_extract->process_from_agg_idxes(start, end, false);
	job->wait_end();

	_nraw_model->set_format(get_format_from_dom());
	_nraw_model->set_nraw(*_nraw);
	_nraw_model->set_starting_row(start);
	_nraw_widget->resize_columns_content();
	_nraw_model->set_invalid_elements(job->get_invalid_evts());

	// Set the invalid lines widget
	_inv_lines_widget->clear();
	for (auto const& line : job->get_invalid_evts()) {
		_inv_lines_widget->addItem(QString::fromStdString(line.second));
		_nraw_widget->mark_row_as_invalid(line.first);
	}
}

void App::PVFormatBuilderWidget::slotExtractorPreview()
{
	if (_log_input_type) {
		PVRow start, end;
		_nraw_widget->get_ext_args(start, end);
		load_log(start, end);
	}
}

bool App::PVFormatBuilderWidget::is_dom_empty()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return !rootDom.hasChildNodes();
}

void App::PVFormatBuilderWidget::slotItemClickedInView(const QModelIndex& index)
{
	// Automatically set the good columns in the mini-extractor

	// Get the PVXmlTreeNodeDom object that comes with that index
	PVRush::PVXmlTreeNodeDom* node = myTreeModel->nodeFromIndex(index);

	// If this is the root item, do nothing.
	if (!node || node->getParent() == nullptr) {
		_nraw_widget->unselect_column();
		return;
	}

	// Then, update the linear fields id in PVXmlTreeNode's tree.
	myTreeModel->updateFieldsLinearId();

	// If this is not an axis, no need to highlight a column in the preview listing
	if (node->typeToString() != "axis") {
		_nraw_widget->unselect_column();
	} else {
		node = node->getFirstFieldParent();
		if (node) {
			PVCol field_id = node->getFieldLinearId();
			_nraw_widget->select_column(field_id);
		}
	}
}

void App::PVFormatBuilderWidget::slotItemClickedInMiniExtractor(PVCol column)
{
	/* Automatically selection the good axis in the tree view */

	// Update the linear fields id in PVXmlTreeNode's tree.
	myTreeModel->updateFieldsLinearId();

	QModelIndex index = get_field_node_index(column, myTreeModel->index(0, 0));

	if (index.isValid()) {
		myTreeView->setCurrentIndex(index);
		myParamBord_old_model->edit(index);
		_nraw_widget->select_header(column);
	}
}

QModelIndex App::PVFormatBuilderWidget::get_field_node_index(const PVCol field_id,
                                                                     const QModelIndex& parent)
{
	QModelIndex index = QModelIndex();

	if (!parent.isValid())
		return {};

	int sibling = 0;
	do {
		index = parent.sibling(sibling, 0);
		PVRush::PVXmlTreeNodeDom* node = myTreeModel->nodeFromIndex(index);

		if (node->typeToString() == "axis" &&
		    node->getFirstFieldParent()->getFieldLinearId() == field_id)
			return index;

		index = get_field_node_index(field_id, myTreeModel->index(0, 0, index));
	} while (parent.sibling(sibling++, 0).isValid() && !index.isValid());

	return index;
}

void App::PVFormatBuilderWidget::set_axes_name_selected_row_Slot(int row)
{
	assert(_nraw);
	if ((PVRow)row >= _nraw->row_count()) {
		return;
	}
	QStringList names;
	for (PVCol j(0); j < _nraw->column_count(); j++) {
		// We need to do a deep copy of this
		names << QString::fromStdString(_nraw->at_string(row, j));
	}
	myTreeModel->setAxesNames(names);
}

bool App::PVFormatBuilderWidget::openFormat(QString const& path)
{
	QFile f(path);
	if (f.exists()) { // if the file exists...
		if (myTreeModel->openXml(path)) {
			_options_widget->set_lines_range(myTreeModel->get_first_line(),
			                                 myTreeModel->get_line_count());
			bool is_path, disabled;
			const QString& python_script = myTreeModel->get_python_script(is_path, disabled);
			_options_widget->set_python_script(python_script, is_path, disabled);
			_cur_file = path;
			setWindowTitleForFile(path);
			myTreeView->expandAll();
			return true;
		}
	}

	return false;
}

void App::PVFormatBuilderWidget::openFormat(QDomDocument& doc)
{
	myTreeModel->openXml(doc);
}

void App::PVFormatBuilderWidget::slotMainTabChanged(int idx)
{
	if (idx == 2) {
		// This is the axes combination editor.

		// Get the list of axes and update the axis combination
		myTreeModel->updateAxesCombination();
		_axes_comb_widget->update_all();
	}
}
