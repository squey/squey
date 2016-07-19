/**
 * @file
 *
 * @copyright (C) Picviz Labs 2011-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef PVFORMATBUILDER_H
#define PVFORMATBUILDER_H
#include <iostream>

#include <QTreeView>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QPushButton>
#include <QWidget>
#include <QToolBar>
#include <QFileDialog>
#include <QAction>
#include <QString>
#include <QMenuBar>
#include <QAbstractItemModel>
#include <QMainWindow>
#include <QDomElement>
#include <QDomDocument>
#include <QListWidget>

#include <PVXmlDomModel.h>
#include <PVXmlTreeView.h>
#include <PVXmlParamWidget.h>
#include <PVNrawListingWidget.h>
#include <PVNrawListingModel.h>
#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/core/PVClassLibrary.h>
#include <pvkernel/core/PVArgument.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVInputType.h>

namespace PVGuiQt
{
class PVAxesCombinationWidget;
}

namespace PVInspector
{

class PVOptionsWidget;

class PVFormatBuilderWidget : public QMainWindow
{
	Q_OBJECT
  public:
	PVFormatBuilderWidget(QWidget* parent = nullptr);

	virtual ~PVFormatBuilderWidget();

  private:
	void closeEvent(QCloseEvent* event);
	void init(QWidget* parent = 0);

  public:
	bool openFormat(QString const& path);
	void openFormat(QDomDocument& doc);
	PVRush::types_groups_t& getGroups() { return myTreeModel->getGroups(); }

  private:
	int lastSplitterPluginAdding;

	void actionAllocation();

	/**
	 * initialise les connexions dont tout les emitter/reciever sont des attributs
	 * de la classe
	 */
	void initConnexions();

	/**
	 * init the menubar
	 */
	void initMenuBar();

	void setWindowTitleForFile(QString const& path);

	bool save();
	bool saveAs();

	/**
	 * init the toolsbar
	 * @param vb
	 */
	void initToolBar(QVBoxLayout* vb);

  protected:
	/**
	 * Clear filter data and run extraction filling NRaw and invalid elements.
	 */
	void update_table(PVRow start, PVRow end);

	/**
	 * Get the PVFormat from its dom representation.
	 */
	PVRush::PVFormat get_format_from_dom();

	/**
	 * Stop old extractor, create the new one with default argument and starts it.
	 */
	void create_extractor();

	/**
	 * Try to find a matching splitter when we import a file without format.
	 */
	void guess_first_splitter();
	bool is_dom_empty();

  private:
	void load_log(PVRow rstart, PVRow rend);

  public Q_SLOTS:
	// Tree slots
	void slotAddAxisIn();
	void slotAddFilterAfter();
	void slotAddRegExAfter();
	void slotAddSplitter();
	void slotAddConverter();
	void slotAddUrl();
	void slotApplyModification();
	void slotDelete();
	void slotMoveUp();
	void slotMoveDown();
	void slotNeedApply();
	void slotNewWindow();
	QString slotOpen();
	void slotOpenLog();
	void slotSave();
	void slotSaveAs();
	void slotUpdateToolsState(const QModelIndex& index = QModelIndex());
	void slotExtractorPreview();
	void slotItemClickedInView(const QModelIndex& index);
	void slotMainTabChanged(int idx);

	// Slot for the NRAW listing
	void set_axes_name_selected_row_Slot(int row);

  protected:
	PVRush::PVInputDescription_p _log_input; //!< File use for Format building.
	PVRush::PVInputType_p _log_input_type;   //!< InputType plugin to load data.
	PVRush::PVSourceCreator_p _log_sc;       //!< The source from input file.
	PVRush::PVRawSourceBase_p _log_source;
	std::shared_ptr<PVRush::PVExtractor> _log_extract; //!< Extractor to load data.
	PVOptionsWidget* _options_widget;
	PVGuiQt::PVAxesCombinationWidget* _axes_comb_widget;

	// Model and widget for the NRAW
	PVNrawListingModel* _nraw_model;
	PVNrawListingWidget* _nraw_widget;

	// Invalid lines
	QListWidget* _inv_lines_widget;

	static QList<QUrl> _original_shortcuts;

  protected:
	QString _cur_file;

  private:
	PVRush::PVInputType::list_inputs _inputs; //!< List of input files.

	QFileDialog _file_dialog;

	// FIXME: Those variables names are crap!
	PVXmlTreeView* myTreeView;
	PVXmlDomModel* myTreeModel; //!< Model for the Tree representation of the format.
	PVXmlParamWidget* myParamBord_old_model;
	QWidget* myParamBord;
	QWidget emptyParamBoard;
	QTabWidget* _main_tab;
	//
	QVBoxLayout* vbParam;
	QMenuBar* menuBar;
	Inendi::PVSource* _org_source; // If this widget is bound to a PVSource's format

	//
	QFile logFile; ///!< file we open to edit the format

	QMenu* _splitters;
	QMenu* _converters;

	QAction* actionAddAxisAfter;
	QAction* actionAddAxisIn;
	QAction* actionAddFilterAfter;
	QAction* actionAddRegExAfter;
	QAction* actionAddRegExBefore;
	QAction* actionAddUrl;
	QAction* actionAddRegExIn;
	QPushButton* actionApply;
	QAction* actionCloseWindow;
	QAction* actionDelete;
	QAction* actionMoveUp;
	QAction* actionMoveDown;
	QAction* actionNewWindow;
	QAction* actionOpen;
	QAction* actionSave;
	QAction* actionSaveAs;
};
}
#endif /* PVFORMATBUILDER_H */
