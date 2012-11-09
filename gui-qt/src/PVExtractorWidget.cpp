/**
 * \file PVExtractorWidget.cpp
 *
 * Copyright (C) Picviz Labs 2009-2012
 */

#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QStringList>

#include <pvkernel/core/general.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvhive/PVCallHelper.h>
#include <pvhive/PVHive.h>

#include <pvguiqt/PVProjectsTabWidget.h>

#include <PVExtractorWidget.h>

#include <boost/thread.hpp>

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::PVFilterWidget
 *
 *****************************************************************************/
PVInspector::PVExtractorWidget::PVExtractorWidget(Picviz::PVSource& lib_src, PVGuiQt::PVProjectsTabWidget* projects_tab, QWidget* parent):
	QDialog(parent),
	_lib_src(&lib_src),
	_projects_tab(projects_tab)
{
	_batch_size = lib_src.get_extraction_last_nlines();
	_slider_pressed_value = 0;

	//VARIABLES
	QPushButton *exit = new QPushButton("Exit", this);
	QPushButton *read_all = new QPushButton("Update number of lines of inputs", this);
	QPushButton *process = new QPushButton("Process input", this);
	QVBoxLayout *main_layout    = new QVBoxLayout();
	QHBoxLayout *buttons_layout = new QHBoxLayout();
	QHBoxLayout *top_layout = new QHBoxLayout();
	QVBoxLayout* left_layout = new QVBoxLayout();
	QGridLayout *grid_layout = new QGridLayout();
	QSpacerItem* spacer_v1 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
	QSpacerItem* spacer_v2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

	_list_inputs = new QTreeWidget(this);
	_slider_index = new QSlider(Qt::Horizontal,this);
	_size_batch_widget = new QLineEdit(this);
	_source_starts_filename = new QLabel();
	_source_starts_directory = new QLabel();
	_source_starts_line = new QLineEdit();
	_sources_number_lines = new QLineEdit();
	_sources_number_lines->setReadOnly(true);

	// Set the grid layout
	grid_layout->addWidget(new QLabel("Extract from:"),0,0);
	grid_layout->addWidget(_slider_index,0,2);
	grid_layout->addWidget(new QLabel("Number of lines:"),1,0);
	grid_layout->addWidget(_size_batch_widget,1,2);
	grid_layout->addWidget(new QLabel("Extraction will start from"),2,0);
	grid_layout->addWidget(_source_starts_filename,2,2);
	grid_layout->addWidget(new QLabel("in directory"),3,0);
	grid_layout->addWidget(_source_starts_directory,3,2);
	grid_layout->addWidget(new QLabel("from line"),4,0);
	grid_layout->addWidget(_source_starts_line,4,2);
	grid_layout->addItem(spacer_v1,5,0);
	grid_layout->addItem(spacer_v2,5,2);
	grid_layout->setColumnMinimumWidth(1, 10);
	grid_layout->setVerticalSpacing(16);

	// Set the left layout
	left_layout->addWidget(_list_inputs);
	QHBoxLayout* lines_layout = new QHBoxLayout();
	lines_layout->addWidget(new QLabel("Total known number of lines:"));
	lines_layout->addWidget(_sources_number_lines);
	left_layout->addItem(lines_layout);

	// Add the widgets to the layouts
	main_layout->addItem(top_layout);	
	  top_layout->addItem(left_layout);
	  top_layout->addItem(grid_layout);
	main_layout->addItem(buttons_layout);	
	  buttons_layout->addWidget(read_all);
	  buttons_layout->addWidget(process);
	  buttons_layout->addWidget(exit);
	
	// Init the source list
	_list_inputs->setColumnCount(2);
	_list_inputs->setHeaderLabels(QStringList() << "Source" << "Known number of lines");
	update_scroll();

	// Fill the source list
	fill_source_list();

	// Init the slider and other infos
	_slider_index->setMinimum(0);
	_slider_index->setValue(lib_src.get_extraction_last_start());
	_slider_index->setTickPosition(QSlider::TicksBelow);
	QIntValidator *iv = new QIntValidator();
	iv->setBottom(10);
	iv->setTop(PICVIZ_LINES_MAX);
	_size_batch_widget->setValidator(iv);
	_size_batch_widget->setText(QString("%1").arg(_batch_size));
	QIntValidator *iv_line = new QIntValidator();
	iv_line->setBottom(0);
	_source_starts_line->setValidator(iv_line);

	connect(exit, SIGNAL(pressed()), this, SLOT(exit_Slot()));
	connect(read_all, SIGNAL(pressed()), this, SLOT(read_all_Slot()));
	connect(process, SIGNAL(pressed()), this, SLOT(process_Slot()));
	connect(_slider_index, SIGNAL(valueChanged(int)), this, SLOT(slider_change_Slot(int)));
	connect(_slider_index, SIGNAL(sliderPressed()), this, SLOT(slider_pressed_Slot()));
	//connect(_slider_index, SIGNAL(sliderReleased()), this, SLOT(slider_released_Slot()));
	connect(_size_batch_widget, SIGNAL(textEdited(QString const&)), this, SLOT(size_batch_edited_Slot(QString const&)));
	connect(_source_starts_line, SIGNAL(textEdited(QString const&)), this, SLOT(line_start_edited_Slot(QString const&)));

	get_extractor().get_agg().debug();

	update_infos();

	setLayout(main_layout);
	setWindowTitle(QString("Picviz source extractor: ") + this->lib_src().get_window_name());
}

void PVInspector::PVExtractorWidget::refresh_and_show()
{
	fill_source_list();
	update_scroll();
	update_infos();
	show();
}

void PVInspector::PVExtractorWidget::fill_source_list()
{
	_list_inputs->clear();
	_total_nlines = 0;
	PVRush::PVAggregator::list_inputs::const_iterator it;
	const PVRush::PVAggregator::list_inputs& in = get_extractor().get_inputs();
	QList<QTreeWidgetItem *> items;
	for (it = in.begin(); it != in.end(); it++) {
		size_t nlines = (*it)->last_elt_index();
		if (nlines > 0) {
			nlines++;
			_total_nlines += nlines;
		}
		items.append(new QTreeWidgetItem((QTreeWidget*)NULL, QStringList() << (*it)->human_name() << QString("%1").arg(nlines)));
	}
	_list_inputs->insertTopLevelItems(0, items);
	_slider_index->setMaximum(_total_nlines-1);
	_sources_number_lines->setText(QString::number(_total_nlines));
}

void PVInspector::PVExtractorWidget::update_status_ext(PVCore::PVProgressBox* pbox, PVRush::PVControllerJob_p job)
{
	while (job->running()) {
		pbox->set_status(job->status());
		pbox->set_extended_status(QString("Number of rejected elements: %L1").arg(job->rejected_elements()));
		boost::this_thread::sleep(boost::posix_time::milliseconds(200));
	}
}
bool PVInspector::PVExtractorWidget::process_extraction_job(PVRush::PVControllerJob_p job)
{
	bool ret = true;
	PVRush::PVExtractor& ext = get_extractor();
	// Show a progress box that will finish with "accept" when the job is done
	if (!PVExtractorWidget::show_job_progress_bar(job, ext.get_format().get_format_name(), job->nb_elts_max(), this)) {
		//ext.restore_nraw();
		ret = false;
	}
	else {
		lib_src().wait_extract_end(job);
		if (ext.get_nraw().get_number_rows() == 0) {
			// Empty extraction, cancel it.
			QMessageBox::warning(this, tr("Empty extraction"), tr("The extraction just performed is empty. Returning to the previous state..."));
			//ext.restore_nraw();
			ret = false;
		}
		else {
			//ext.clear_saved_nraw();
		}
	}

	return ret;
}

bool PVInspector::PVExtractorWidget::show_job_progress_bar(PVRush::PVControllerJob_p job, QString const& desc, int nlines, QWidget* parent = NULL)
{
	PVCore::PVProgressBox *pbox = new PVCore::PVProgressBox(tr("Extracting %1...").arg(desc), parent, 0, QString("Number of elements processed: %L1/%L2"));
	QProgressBar *pbar = pbox->getProgressBar();
	pbar->setValue(0);
	pbar->setMaximum(nlines);
	pbar->setMinimum(0);
	connect(job.get(), SIGNAL(job_done_signal()), pbox, SLOT(accept()));
	// launch a thread in order to update the status of the progress bar
	boost::thread th_status(boost::bind(update_status_ext, pbox, job));	
	pbox->launch_timer_status();
	if (!job->running() && (job->started())) {
		return true;
	}
	if (pbox->exec() == QDialog::Accepted) {
		return true;
	}

	// Cancel this job and ask the user if he wants to keep the extracted data.
	job->cancel();
	PVLOG_DEBUG("extractor: job canceled !\n");
	QMessageBox msgbox(QMessageBox::Question, tr("Extraction of %1 canceled").arg(desc), tr("Extraction has been canceled. Do you want to proceed with the %1 lines that have been processed ?").arg(job->status()), QMessageBox::Yes | QMessageBox::No, parent);
	return msgbox.exec() == QMessageBox::Yes;
}

void PVInspector::PVExtractorWidget::process_Slot()
{
    PVLOG_DEBUG("PVInspector::PVExtractorWidget::process_Slot()\n");

	size_t index = _slider_index->value();
	_batch_size = _size_batch_widget->text().toLong();
	
	//get_extractor().save_nraw();
	Picviz::PVSource_sp src_clone = lib_src().clone_with_no_process();
	PVRush::PVExtractor& ext = src_clone->get_extractor();

	PVRush::PVControllerJob_p job = src_clone->extract_from_agg_nlines(index, _batch_size);
	bool success = PVExtractorWidget::show_job_progress_bar(job, ext.get_format().get_format_name(), job->nb_elts_max(), this);

	fill_source_list();

	if (success) {
		//Picviz::PVSource_sp src = lib_src().shared_from_this();
		//PVHive::call<FUNC(Picviz::PVSource::process_from_source)>(src);
		//_view->last_extractor_batch_size = _batch_size;
		PVHive::call<FUNC(Picviz::PVSource::create_default_view)>(src_clone);
		_projects_tab->add_source(src_clone.get());
	}
	else {
		src_clone->remove_from_tree();
	}
}

void PVInspector::PVExtractorWidget::exit_Slot()
{
	done(0);
}

void PVInspector::PVExtractorWidget::slider_change_Slot(int /*value*/)
{
	update_infos();
}

void PVInspector::PVExtractorWidget::slider_pressed_Slot()
{
	_slider_pressed_value = _slider_index->value();
}

void PVInspector::PVExtractorWidget::slider_released_Slot()
{
	if (_slider_index->value() == _slider_pressed_value)
		return;

	process_Slot();
}

void PVInspector::PVExtractorWidget::read_all_Slot()
{
	PVRush::PVControllerJob_p job = get_extractor().read_everything(0);

	PVCore::PVProgressBox *pbox = new PVCore::PVProgressBox(tr("Counting elements..."), this);
	connect(job.get(), SIGNAL(job_done_signal()), pbox, SLOT(accept()));
	if (!job->running() && (job->started())) {
		fill_source_list();
		return;
	}
	if (pbox->exec() != QDialog::Accepted) {
		job->cancel();
	}

	fill_source_list();
}

void PVInspector::PVExtractorWidget::update_scroll()
{
	_slider_index->setPageStep(_batch_size);
	_slider_index->setTickInterval(_batch_size);
}

void PVInspector::PVExtractorWidget::update_infos()
{
	size_t index = _slider_index->value();
	chunk_index offset = 0;
	PVRush::PVRawSourceBase_p src = get_extractor().get_agg().agg_index_to_source(index, &offset);
	_cur_src = src;
	_cur_src_offset = offset;
	QString file;
	if (src == NULL)
	{
		// Take the last one
		file = _list_inputs->takeTopLevelItem(_list_inputs->topLevelItemCount()-1)->text(0);
	}
	else
		file = src->human_name();

	QFileInfo fi(file);
	_source_starts_filename->setText(fi.fileName());
	_source_starts_directory->setText(fi.canonicalPath());
	_source_starts_line->setText(QString("%1").arg(index-offset));
}

void PVInspector::PVExtractorWidget::size_batch_edited_Slot(const QString& text)
{
	_batch_size = text.toULong();
	update_scroll();
}

void PVInspector::PVExtractorWidget::line_start_edited_Slot(const QString &text)
{
	size_t new_index = text.toLong()+_cur_src_offset;
	_slider_index->setValue(new_index);
	//update_infos();
}
