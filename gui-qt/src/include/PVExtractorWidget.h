/**
 * \file PVExtractorWidget.h
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#ifndef PVEXTRACTORWIDGET_H
#define PVEXTRACTORWIDGET_H

#include <QDialog>
#include <QGridLayout>
#include <QTreeWidget>
#include <QSlider>
#include <QLineEdit>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>
#include <QProgressBar>

#include <pvkernel/rush/PVExtractor.h>
#include <pvkernel/rush/PVRawSourceBase_types.h>


#include <picviz/general.h>
#include <picviz/PVSource.h>

namespace PVCore {
class PVProgressBox;
}

namespace PVInspector {

class PVTabSplitter;

/**
 * \class PVExtractorWidget
 */
class PVExtractorWidget : public QDialog
{
	Q_OBJECT

public:
	/**
	 * Constructor.
	 */
	PVExtractorWidget(Picviz::PVSource& lib_src, QWidget* parent = NULL);

	void refresh_and_show();
	static void update_status_ext(PVCore::PVProgressBox* pbox, PVRush::PVControllerJob_p job);
	static bool show_job_progress_bar(PVRush::PVControllerJob_p job, QString const& desc, int nlines, QWidget* parent);

private:
	
	QTreeWidget* _list_inputs;
	QSlider* _slider_index;
	QLineEdit* _size_batch_widget;
	QLabel* _source_starts_filename;
	QLabel* _source_starts_directory;
	QComboBox *_source_starts_sel;
	QLineEdit *_source_starts_line;
	PVRush::PVRawSourceBase_p _cur_src;

public slots:
	void exit_Slot();
	void read_all_Slot();
	void process_Slot();
	void slider_change_Slot(int value);
	void slider_pressed_Slot();
	void slider_released_Slot();
	void size_batch_edited_Slot(const QString& text);
	void line_start_edited_Slot(const QString &text);

protected:
	void fill_source_list();
	void update_scroll();
	void update_infos();
	size_t _total_nlines;
	size_t _batch_size;
	size_t _cur_src_offset;

private:
	inline Picviz::PVSource& lib_src() { return *_lib_src; }
	inline PVRush::PVExtractor& get_extractor() { return lib_src().get_extractor(); }

	bool process_extraction_job(PVRush::PVControllerJob_p job);

private:
	Picviz::PVSource* _lib_src;
	int _slider_pressed_value;
	QLineEdit* _sources_number_lines;
};

}

#endif



