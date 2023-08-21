/* * MIT License
 *
 * © ESI Group, 2015
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 *
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 *
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
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
#include <pvkernel/widgets/PVFileDialog.h>
#include <pvkernel/filter/PVFieldsFilterParamWidget.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>
#include <pvkernel/rush/PVSourceCreator.h>
#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVInputType.h>
#include "pvkernel/rush/PVTypesDiscoveryOutput.h"

namespace Squey
{
class PVSource;
} // namespace Squey

namespace PVGuiQt
{
class PVAxesCombinationWidget;
} // namespace PVGuiQt

namespace App
{

class PVOptionsWidget;

class PVFormatBuilderWidget : public QMainWindow
{
	Q_OBJECT
  public:
	PVFormatBuilderWidget(QWidget* parent = nullptr);

	~PVFormatBuilderWidget() override;

  private:
	void closeEvent(QCloseEvent* event) override;
	void init(QWidget* parent = nullptr);

  public:
	bool openFormat(QString const& path);
	void openFormat(QDomDocument& doc);

	QString get_current_format_name() const { return _cur_file; }

	PVRush::PVFormat load_log_and_guess_format(const PVRush::PVInputDescription_p input,
	                                           const PVRush::PVInputType_p& input_type);

	/**
	 * Get the PVFormat from its dom representation.
	 */
	PVRush::PVFormat get_format_from_dom() const;

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
	void check_for_new_time_formats();

	bool check_format_validity();

	/**
	 * init the toolsbar
	 * @param vb
	 */
	void initToolBar(QVBoxLayout* vb);

	/**
	 * Get the node's index at the given field_id
	 * @param field_id
	 * @param parent
	 * @return index
	 */
	QModelIndex get_field_node_index(const PVCol field_id, const QModelIndex& parent);

  protected:
	/**
	 * Clear filter data and run extraction filling NRaw and invalid elements.
	 */
	void update_table(PVRow start, PVRow end);

	/**
	 * Try to find a matching splitter when we import a file without format.
	 */
	void guess_first_splitter();
	bool is_dom_empty();

  private:
	void load_log(PVRow rstart, PVRow rend);
	void update_types_autodetection_count(const PVRush::PVFormat& format);

	void get_source_creator_from_inputs(const PVRush::PVInputDescription_p input,
	                                    const PVRush::PVInputType_p& input_type,
	                                    PVRush::PVSourceCreator_p& source_creator,
	                                    PVRush::PVRawSourceBase_p& raw_source_base) const;

	PVRush::PVFormat guess_format(const PVRush::PVRawSourceBase_p& raw_source_base,
	                              PVXmlDomModel& tree_model) const;

	PVRush::PVFormat guess_format(const PVRush::PVInputDescription_p input,
	                              const PVRush::PVInputType_p& input_type) const;

  public Q_SLOTS:
	// Tree slots
	void slotAddAxisIn();
	void slotAddFilterAfter();
	void slotSetAxesName();
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
	void slotAutoDetectAxesTypes(bool handle_header = true);
	void slotUpdateToolsState(const QModelIndex& index = QModelIndex());
	void slotExtractorPreview();
	void slotItemClickedInView(const QModelIndex& index);
	void slotItemClickedInMiniExtractor(PVCol column);
	void slotMainTabChanged(int idx);

	// Slot for the NRAW listing
	void set_axes_name_selected_row_Slot(int row);

  protected:
	PVRush::PVInputDescription_p _log_input; //!< File use for Format building.
	PVRush::PVInputType_p _log_input_type;   //!< InputType plugin to load data.
	PVRush::PVSourceCreator_p _log_sc;       //!< The source from input file.
	PVRush::PVRawSourceBase_p _log_source;
	std::unique_ptr<PVRush::PVNraw> _nraw;
	std::unique_ptr<PVRush::PVNrawOutput> _nraw_output;
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

	PVWidgets::PVFileDialog _file_dialog;

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
	Squey::PVSource* _org_source; // If this widget is bound to a PVSource's format

	//
	QFile logFile; ///!< file we open to edit the format

	QMenu* _splitters;
	QMenu* _converters;

	QAction* actionAddAxisAfter;
	QAction* actionAddAxisIn;
	QAction* actionNameAxes;
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
} // namespace App
#endif /* PVFORMATBUILDER_H */
