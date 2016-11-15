/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#include <QSplitter>
#include <QDesktopServices>
#include <QDesktopWidget>

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

#include <pvguiqt/PVAxesCombinationWidget.h>
#include <pvkernel/core/PVRecentItemsManager.h>

#include <boost/thread.hpp>

QList<QUrl> PVInspector::PVFormatBuilderWidget::_original_shortcuts = QList<QUrl>();

#define FORMAT_BUILDER_TITLE (QObject::tr("Format builder"))
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::PVFormatBuilderWidget
 *
 *****************************************************************************/
PVInspector::PVFormatBuilderWidget::PVFormatBuilderWidget(QWidget* parent)
    : QMainWindow(parent), _file_dialog(this)
{
	init(parent);
	setObjectName("PVFormatBuilderWidget");
	setAttribute(Qt::WA_DeleteOnClose, true);
}

void PVInspector::PVFormatBuilderWidget::closeEvent(QCloseEvent* event)
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

void PVInspector::PVFormatBuilderWidget::init(QWidget* /*parent*/)
{
	setWindowTitle(FORMAT_BUILDER_TITLE);

	auto main_splitter = new QSplitter(Qt::Vertical);
	/*
	 * ****************************************************************************
	 * Création of graphics elements.
	 * ****************************************************************************
	 */
	auto vb = new QVBoxLayout();
	vb->setMargin(0);
	auto vertical_splitter = new QSplitter(Qt::Horizontal);
	vbParam = new QVBoxLayout();
	vbParam->setSizeConstraint(QLayout::SetMinimumSize);

	// initialisation of the toolbar.
	actionAllocation();

	menuBar = new QMenuBar();
	initMenuBar();

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
	_nraw_widget->connect_autodetect(this, &PVFormatBuilderWidget::slotAutoDetectAxesTypes);
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
	main_layout->setMenuBar(menuBar);

	auto central_widget = new QWidget();
	central_widget->setLayout(main_layout);

	setCentralWidget(central_widget);

	/* add of user/inendi inspector's path for formats
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
	QRect dr = QDesktopWidget().screenGeometry();
	resize(dr.width() * 0.75, dr.height() * 0.75);
}
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::~PVFormatBuilderWidget
 *
 *****************************************************************************/
PVInspector::PVFormatBuilderWidget::~PVFormatBuilderWidget()
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
 * PVInspector::PVFormatBuilderWidget::actionAllocation
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::actionAllocation()
{
	actionAddAxisIn = new QAction("add an axis", (QObject*)this);
	actionAddAxisIn->setIcon(QIcon(":/add-axis"));
	actionAddFilterAfter = new QAction("add a filter", (QObject*)this);
	actionAddFilterAfter->setIcon(QIcon(":/filter"));
	actionAddRegExAfter = new QAction("add a RegEx", (QObject*)this);
	actionAddRegExAfter->setIcon(QIcon(":/add-regexp"));
	actionAddUrl = new QAction("add URL splitter", (QObject*)this);

	actionSave = new QAction("&Save format", (QObject*)this);
	actionSave->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_S));
	actionSave->setIcon(QIcon(":/save"));
	actionSaveAs = new QAction("Save format as...", (QObject*)this);
	actionDelete = new QAction("Delete", (QObject*)this);
	actionDelete->setShortcut(QKeySequence(Qt::Key_Delete));
	actionDelete->setIcon(QIcon(":/red-cross"));
	actionDelete->setEnabled(false);
	actionMoveDown = new QAction("move down", (QObject*)this);
	actionMoveDown->setShortcut(QKeySequence(Qt::Key_Down));
	actionMoveDown->setIcon(QIcon(":/go-down.png"));
	actionMoveUp = new QAction("move up", (QObject*)this);
	actionMoveUp->setShortcut(QKeySequence(Qt::Key_Up));
	actionMoveUp->setIcon(QIcon(":/go-up.png"));
	actionOpen = new QAction(tr("Open format..."), (QObject*)this);
	actionOpen->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_O));
	actionOpen->setIcon(QIcon(":/document-open.png"));

	actionNewWindow = new QAction(tr("New window"), (QObject*)this);
	actionCloseWindow = new QAction(tr("Close window"), (QObject*)this);
	actionCloseWindow->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_W));
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::initConnexions
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initConnexions()
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

	// Connections for the axes combination editor
	connect(_main_tab, &QTabWidget::currentChanged, this,
	        &PVFormatBuilderWidget::slotMainTabChanged);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::initToolBar
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initToolBar(QVBoxLayout* vb)
{

	auto tools = new QToolBar();

	tools->addAction(actionAddFilterAfter);
	tools->addAction(actionAddAxisIn);

	tools->addSeparator();
	tools->addAction(actionMoveDown);
	tools->addAction(actionMoveUp);
	tools->addSeparator();
	tools->addAction(actionDelete);
	tools->addSeparator();
	tools->addAction(actionOpen);
	tools->addAction(actionSave);

	vb->addWidget(tools);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddAxisIn
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddAxisIn()
{
	myTreeView->addAxisIn();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddFilterAfter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddFilterAfter()
{
	myTreeView->addFilterAfter();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddRegExAfter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddRegExAfter()
{
	myTreeView->addRegExIn();
}
/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddSplitter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddSplitter()
{
	QAction* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	PVFilter::PVFieldsSplitterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_class_by_name(itype);
	PVFilter::PVFieldsSplitterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsSplitterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	PVLOG_DEBUG("(PVInspector::PVFormatBuilderWidget::slotAddSplitter) type_name %s, %s\n",
	            qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
	myTreeView->addSplitter(in_t_cpy);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddConverter
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddConverter()
{
	QAction* action_src = (QAction*)sender();
	QString const& itype = action_src->data().toString();
	PVFilter::PVFieldsConverterParamWidget_p in_t =
	    LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_class_by_name(itype);
	PVFilter::PVFieldsConverterParamWidget_p in_t_cpy =
	    in_t->clone<PVFilter::PVFieldsConverterParamWidget>();
	QString registered_name = in_t_cpy->registered_name();
	PVLOG_DEBUG("(PVInspector::PVFormatBuilderWidget::slotAddConverter) type_name %s, %s\n",
	            qPrintable(in_t_cpy->type_name()), qPrintable(registered_name));
	myTreeView->addConverter(in_t_cpy);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotAddUrl
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotAddUrl()
{
	myTreeView->addUrlIn();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotApplyModification
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotApplyModification()
{
	QModelIndex index;
	myTreeView->applyModification(myParamBord_old_model, index);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotDelete
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotDelete()
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
	}
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotMoveUp
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotMoveUp()
{
	myTreeView->moveUp();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotMoveDown
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotMoveDown()
{
	myTreeView->moveDown();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotNeedApply
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotNeedApply()
{
	QModelIndex index;
	myTreeView->applyModification(myParamBord_old_model, index);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotNewWindow
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotNewWindow()
{
	auto other = new PVFormatBuilderWidget;
	other->move(x() + 40, y() + 40);
	other->show();
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotOpen
 *
 *****************************************************************************/
QString PVInspector::PVFormatBuilderWidget::slotOpen()
{
	_file_dialog.setWindowTitle("Load format from...");
	_file_dialog.setAcceptMode(QFileDialog::AcceptOpen);

	if (!_file_dialog.exec()) {
		return QString();
	}

	const QString urlFile = _file_dialog.selectedFiles().at(0);

	if (urlFile.isEmpty() || (openFormat(urlFile) == false)) {
		return QString();
	}

	return urlFile;
}

bool PVInspector::PVFormatBuilderWidget::save()
{
	// Take focus, so any currently edited argument will be set
	setFocus(Qt::MouseFocusReason);

	if (_cur_file.isEmpty()) {
		return saveAs();
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

void PVInspector::PVFormatBuilderWidget::check_for_new_time_formats()
{
	const std::unordered_set<std::string> time_formats = get_format_from_dom().get_time_formats();
	std::unordered_set<std::string> supported_time_formats =
	    PVRush::PVTypesDiscoveryOutput::supported_time_formats();
	std::unordered_set<std::string> new_time_formats;

	for (const std::string& time_format : time_formats) {
		auto it = supported_time_formats.find(time_format);
		if (it == supported_time_formats.end() && time_format.find("epoch") == std::string::npos) {
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
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotSave
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotSave()
{
	save();
}

bool PVInspector::PVFormatBuilderWidget::saveAs()
{
	setFocus(Qt::MouseFocusReason);

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
 * PVInspector::PVFormatBuilderWidget::slotSaveAs
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotSaveAs()
{
	saveAs();
}

void PVInspector::PVFormatBuilderWidget::slotAutoDetectAxesTypes()
{
	static constexpr const size_t mega = 1024 * 1024;

	PVRow start, end;
	_nraw_widget->get_autodetect_args(start, end);

	bool is_row_count_known = end != 0;
	if (not is_row_count_known) {
		end = EXTRACTED_ROW_COUNT_LIMIT;
	}

	PVCore::PVProgressBox::progress(
	    [&](PVCore::PVProgressBox& pbox) {
		    PVRush::PVTypesDiscoveryOutput type_discovery_output;
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

		    // TODO : it would be nice to ignore axes set by user...

		    // Update format with discovered axes types
		    QDomElement& dom = myTreeModel->getRootDom();
		    QDomNodeList axes = dom.elementsByTagName("axis");
		    for (PVCol i(0); i < axes.size(); i++) {
			    QDomElement ax = axes.at(i).toElement();
			    std::string type;
			    std::string type_format;
			    std::tie(type, type_format) = type_discovery_output.type_desc(i);
			    ax.setAttribute("type", type.c_str());
			    ax.setAttribute("type_format", type_format.c_str());
		    }

		    QMetaObject::invokeMethod(this, "slotExtractorPreview");
		},
	    QObject::tr("Autodetecting axes types..."), nullptr);
}

void PVInspector::PVFormatBuilderWidget::update_types_autodetection_count(
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

void PVInspector::PVFormatBuilderWidget::setWindowTitleForFile(QString const& path)
{
	// Change the window title with the filename of the format
	QFileInfo fi(path);
	setWindowTitle(FORMAT_BUILDER_TITLE + QString(" - ") + fi.fileName());
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotUpdateToolsState
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotUpdateToolsState(const QModelIndex& index)
{
	PVRush::PVXmlTreeNodeDom* node = myTreeModel->nodeFromIndex(index);

	bool force_root_action = not index.parent().isValid();

	if (node->getDom().tagName() == "field") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(true);
		actionAddAxisIn->setEnabled(true);
		_splitters->setEnabled(true);
		_converters->setEnabled(true);
		actionDelete->setEnabled(false);
	} else if (node->getDom().tagName() == "axis") {
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(false);
		_converters->setEnabled(false);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "filter" || node->getDom().tagName() == "converter") {
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(force_root_action);
		_converters->setEnabled(force_root_action);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "splitter") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(force_root_action);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(false);
		_converters->setEnabled(force_root_action);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "RegEx") {
		myTreeView->expandRecursive(index);
		actionAddFilterAfter->setEnabled(false);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(false);
		_converters->setEnabled(false);
		actionDelete->setEnabled(true);
	} else if (node->getDom().tagName() == "url") {
		actionAddFilterAfter->setEnabled(false);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(false);
		_converters->setEnabled(false);
		actionDelete->setEnabled(true);
	} else {
		actionAddFilterAfter->setEnabled(true);
		actionAddAxisIn->setEnabled(false);
		_splitters->setEnabled(true);
		_converters->setEnabled(true);
		actionDelete->setEnabled(false);
	}
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::initMenuBar
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::initMenuBar()
{
	QMenu* file = menuBar->addMenu(tr("&File"));

	file->addAction(actionNewWindow);
	file->addSeparator();
	file->addAction(actionOpen);
	file->addAction(actionSave);
	file->addAction(actionSaveAs);
	file->addSeparator();
	PVGuiQt::PVInputTypeMenuEntries::add_inputs_to_menu(file, this, SLOT(slotOpenLog()));
	file->addSeparator();
	file->addAction(actionCloseWindow);

	// add all splitting plugins
	_splitters = menuBar->addMenu(tr("&Splitters"));

	for (const auto it : LIB_CLASS(PVFilter::PVFieldsSplitterParamWidget)::get().get_list()) {
		PVFilter::PVFieldsSplitterParamWidget_p pluginsSplitter = it.value();
		assert(pluginsSplitter);
		QAction* action = pluginsSplitter->get_action_menu(this);

		if (action) {
			action->setData(it.key());
			connect(action, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddSplitter);
			_splitters->addAction(action);
		}
	}

	_splitters->addAction(actionAddUrl);

	// add all conversion plugins
	_converters = menuBar->addMenu(tr("&Converters"));

	for (const auto it : LIB_CLASS(PVFilter::PVFieldsConverterParamWidget)::get().get_list()) {
		PVFilter::PVFieldsConverterParamWidget_p pluginsConverter = it.value();
		assert(pluginsConverter);
		QAction* action = pluginsConverter->get_action_menu(this);

		if (action) {
			action->setData(it.key());
			connect(action, &QAction::triggered, this, &PVFormatBuilderWidget::slotAddConverter);
			_converters->addAction(action);
		}
	}
}

void PVInspector::PVFormatBuilderWidget::get_source_creator_from_inputs(
    const PVRush::PVInputDescription_p input,
    const PVRush::PVInputType_p& input_type,
    PVRush::PVSourceCreator_p& source_creator,
    PVRush::PVRawSourceBase_p& raw_source_base) const
{
	// Get the first input selected
	PVLOG_DEBUG("Input: %s\n", qPrintable(input_type->human_name_of_input(input)));

	// Get list of inputs from the plugin.
	for (PVRush::PVSourceCreator_p sc :
	     PVRush::PVSourceCreatorFactory::get_by_input_type(input_type)) {
		if (sc->pre_discovery(input)) {
			try {
				source_creator = sc;
				// The moni-extractor use the discovery source, as not that much processing is
				// done (it can be handle locally for instance !)
				raw_source_base = source_creator->create_source_from_input(input);
			} catch (PVRush::PVFormatInvalid& e) {
				source_creator.reset();
				continue;
			} catch (std::ios_base::failure const& e) {
				// File can't be found, looks for another type.
				source_creator.reset();
				continue;
			}
			// If the log_source can't be create, look for another source.
			if (raw_source_base.get() == nullptr) {
				source_creator.reset();
				continue;
			}
			break;
		}
	}
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::guess_format
 *
 *****************************************************************************/
PVRush::PVFormat
PVInspector::PVFormatBuilderWidget::guess_format(const PVRush::PVRawSourceBase_p& raw_source_base,
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

	return PVRush::PVFormat(tree_model.getRootDom(), true);
}

PVRush::PVFormat
PVInspector::PVFormatBuilderWidget::guess_format(const PVRush::PVInputDescription_p input,
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
 * PVInspector::PVFormatBuilderWidget::load_log
 *
 *****************************************************************************/

void PVInspector::PVFormatBuilderWidget::load_log(PVRow rstart, PVRow rend)
{
	// If no files where selected, ask for one.
	if (_inputs.isEmpty()) {

		QString choosenFormat;
		PVRush::hash_formats formats, new_formats;

		// This case is only encountered when a source is loaded from the menu
		PVCore::PVArgumentList args;
		if (!_log_input_type->createWidget(formats, new_formats, _inputs, choosenFormat, args,
		                                   this)) {
			return; // This means that the user pressed the "cancel" button
		}
		assert(not _inputs.empty() && "At least one file ahve to be seleced");
	}

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
				    tr("It appears that the %1 splitter can process '%2' and create %3 fields.\n")
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

		_nraw.reset(new PVRush::PVNraw());
		_nraw_output.reset(new PVRush::PVNrawOutput(*_nraw));
		QList<std::shared_ptr<PVRush::PVInputDescription>> list_inputs;
		list_inputs << _log_input;

		_log_extract.reset(new PVRush::PVExtractor(format, *_nraw_output, _log_sc, list_inputs));

		update_table(rstart, rend);

	} catch (PVRush::PVInputException& e) {
		_log_input = PVRush::PVInputDescription_p();
		QMessageBox err(QMessageBox::Critical, tr("Error"),
		                tr("Error while importing a source: %1").arg(QString(e.what())));
		err.show();
		return;
	} catch (PVFilter::PVFieldsFilterInvalidArguments const& e) {
		QMessageBox::critical(this, "Error", e.what());
		return;
	} catch (PVRush::PVFormatInvalid const& e) {
		QMessageBox::critical(this, "Error",
		                      "The current format is not valid. We can't perform an import : " +
		                          QString(e.what()));
		return;
	}

	// Tell the NRAW widget that the input has changed
	_nraw_widget->set_last_input(_log_input_type, _log_input);
}

/******************************************************************************
 *
 * PVInspector::PVFormatBuilderWidget::slotOpenLog
 *
 *****************************************************************************/
void PVInspector::PVFormatBuilderWidget::slotOpenLog()
{
	_log_input_type = PVGuiQt::PVInputTypeMenuEntries::input_type_from_action((QAction*)sender());

	_inputs.clear();

	load_log(FORMATBUILDER_EXTRACT_START_DEFAULT, FORMATBUILDER_EXTRACT_END_DEFAULT);
}

PVRush::PVFormat PVInspector::PVFormatBuilderWidget::get_format_from_dom() const
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return PVRush::PVFormat{rootDom, true};
}

void PVInspector::PVFormatBuilderWidget::update_table(PVRow start, PVRow end)
{
	assert(_log_extract);
	assert(end > start);

	// Clear the filter previous data
	myTreeModel->clearFiltersData();

	// Update the data displaying of the filter param widgers
	myTreeModel->updateFiltersDataDisplay();

	PVRush::PVControllerJob_p job = _log_extract->process_from_agg_idxes(start, end);
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

void PVInspector::PVFormatBuilderWidget::slotExtractorPreview()
{
	PVRow start, end;
	_nraw_widget->get_ext_args(start, end);
	load_log(start, end);
}

bool PVInspector::PVFormatBuilderWidget::is_dom_empty()
{
	QDomElement const& rootDom = myTreeModel->getRootDom();
	return !rootDom.hasChildNodes();
}

void PVInspector::PVFormatBuilderWidget::slotItemClickedInView(const QModelIndex& index)
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

void PVInspector::PVFormatBuilderWidget::slotItemClickedInMiniExtractor(PVCol column)
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

QModelIndex PVInspector::PVFormatBuilderWidget::get_field_node_index(const PVCol field_id,
                                                                     const QModelIndex& parent)
{
	QModelIndex index = QModelIndex();

	if (!parent.isValid())
		return QModelIndex();

	int sibling = 0;
	do {
		index = parent.sibling(sibling, 0);
		PVRush::PVXmlTreeNodeDom* node = myTreeModel->nodeFromIndex(index);

		if (node->typeToString() == "axis" &&
		    node->getFirstFieldParent()->getFieldLinearId() == field_id)
			return index;

		index = get_field_node_index(field_id, index.child(0, 0));
	} while (parent.sibling(sibling++, 0).isValid() && !index.isValid());

	return index;
}

void PVInspector::PVFormatBuilderWidget::set_axes_name_selected_row_Slot(int row)
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

bool PVInspector::PVFormatBuilderWidget::openFormat(QString const& path)
{
	QFile f(path);
	if (f.exists()) { // if the file exists...
		if (myTreeModel->openXml(path)) {
			_options_widget->set_lines_range(myTreeModel->get_first_line(),
			                                 myTreeModel->get_line_count());
			_cur_file = path;
			setWindowTitleForFile(path);
			myTreeView->expandAll();
			return true;
		}
	}

	return false;
}

void PVInspector::PVFormatBuilderWidget::openFormat(QDomDocument& doc)
{
	myTreeModel->openXml(doc);
}

void PVInspector::PVFormatBuilderWidget::slotMainTabChanged(int idx)
{
	if (idx == 2) {
		// This is the axes combination editor.

		// Get the list of axes and update the axis combination
		myTreeModel->updateAxesCombination();
		_axes_comb_widget->update_all();
	}
}
