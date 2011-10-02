//! \file PVFilterWidget.cpp
//! $Id: PVExtractorWidget.cpp 3251 2011-07-06 11:51:57Z rpernaudat $
//! Copyright (C) Sébastien Tricaud 2009-2011
//! Copyright (C) Philippe Saadé 2009-2011
//! Copyright (C) Picviz Labs 2011

#include <QtCore>
#include <QtGui>

#include <QVBoxLayout>
#include <QDialog>
#include <QMessageBox>
#include <QVBoxLayout>
#include <QLabel>
#include <QStringList>

#include <pvkernel/core/general.h>
#include <PVExtractorWidget.h>
#include <PVMainWindow.h>
#include <PVTabSplitter.h>

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/rush/PVRawSourceBase.h>

#include <pvsdk/PVMessenger.h>

#include <tbb/compat/thread>

/******************************************************************************
 *
 * PVInspector::PVFilterWidget::PVFilterWidget
 *
 *****************************************************************************/
PVInspector::PVExtractorWidget::PVExtractorWidget(PVTabSplitter* parent_tab) :
	QDialog((QWidget*)parent_tab),
	_ext(parent_tab->get_lib_view()->get_extractor())
{
	main_window = parent_tab->get_main_window();
	_view = parent_tab->get_lib_view();
	_inspector_tab = parent_tab;
	_batch_size = _view->last_extractor_batch_size;
	_slider_pressed_value = 0;

	//VARIABLES
	QPushButton *exit = new QPushButton("Exit", this);
	QPushButton *read_all = new QPushButton("Update number of lines of inputs", this);
	QPushButton *process = new QPushButton("Process input", this);
	QVBoxLayout *main_layout    = new QVBoxLayout();
	QHBoxLayout *buttons_layout = new QHBoxLayout();
	QHBoxLayout *top_layout = new QHBoxLayout();
	QGridLayout *grid_layout = new QGridLayout();
	QSpacerItem* spacer_v1 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
	QSpacerItem* spacer_v2 = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

	_list_inputs = new QTreeWidget(this);
	_slider_index = new QSlider(Qt::Horizontal,this);
	_size_batch_widget = new QLineEdit(this);
	_source_starts_filename = new QLabel();
	_source_starts_directory = new QLabel();
	_source_starts_line = new QLineEdit();

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

	// Add the widgets to the layouts
	main_layout->addItem(top_layout);	
	  top_layout->addWidget(_list_inputs);
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

	_ext.get_agg().debug();

	update_infos();

	setLayout(main_layout);
	setWindowTitle(QString("Picviz source extractor: ") + parent_tab->get_tab_name());
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
	const PVRush::PVAggregator::list_inputs& in = _ext.get_inputs();
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
}

void PVInspector::PVExtractorWidget::update_status_ext(PVCore::PVProgressBox* pbox, PVRush::PVControllerJob_p job)
{
	while (job->running()) {
		pbox->set_status(job->status());
		std::this_thread::sleep_for(tbb::tick_count::interval_t(0.2));
	}
}

bool PVInspector::PVExtractorWidget::show_job_progress_bar(PVRush::PVControllerJob_p job, QString const& desc, int nlines, QWidget* parent = NULL)
{
	PVCore::PVProgressBox *pbox = new PVCore::PVProgressBox(tr("Extracting %1...").arg(desc), parent, 0, QString("Number of elements processed: %1/%2"));
	QProgressBar *pbar = pbox->getProgressBar();
	pbar->setValue(0);
	pbar->setMaximum(nlines);
	pbar->setMinimum(0);
	connect(job.get(), SIGNAL(job_done_signal()), pbox, SLOT(accept()));
	// launch a thread in order to update the status of the progress bar
	std::thread th_status(boost::bind(update_status_ext, pbox, job));	
	pbox->launch_timer_status();
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
	// The nraw will be updated, view won't be consistent during this process
	_view->set_consistent(false);

	size_t index = _slider_index->value();
	_batch_size = _size_batch_widget->text().toLong();

    PVLOG_DEBUG("PVInspector::PVExtractorWidget::process_Slot() l:%d\n",__LINE__);
	_ext.save_nraw();
    PVLOG_DEBUG("PVInspector::PVExtractorWidget::process_Slot() l:%d\n",__LINE__);
	PVRush::PVControllerJob_p job = _ext.process_from_agg_nlines(index, _batch_size, 0);
    PVLOG_DEBUG("PVInspector::PVExtractorWidget::process_Slot() l:%d\n",__LINE__);

	// Show a progress box that will finish with "accept" when the job is done
	if (!show_job_progress_bar(job, _ext.get_format().get_format_name(), _batch_size, this)) {
		_ext.restore_nraw();
		_view->set_consistent(true);
		PVSDK::PVMessage message;
		message.function = PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW;
		message.pv_view = _view;
		main_window->get_pvmessenger()->post_message_to_gl(message);
		return;
	}
	_ext.clear_saved_nraw();

	PVLOG_INFO("extractor: the normalization job took %0.4f seconds.\n", job->duration().seconds());

	// Recreate mapping and plotting from source
	_view->recreate_mapping_plotting();
	
	// Destroy all layers, and recreate the first one
	_view->reset_layers();
	
	_inspector_tab->refresh_layer_stack_view_Slot();
	_inspector_tab->refresh_listing_Slot();

	// Now view is consistent
	_view->set_consistent(true);

	fill_source_list();

	_view->last_extractor_batch_size = _batch_size;
		
	// Send a message to PVGL
	PVSDK::PVMessage message;
	message.function = PVSDK_MESSENGER_FUNCTION_REINIT_PVVIEW;
	message.pv_view = _view;
	main_window->get_pvmessenger()->post_message_to_gl(message);
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
	PVRush::PVControllerJob_p job = _ext.read_everything(0);
	job->wait_end();

	// Update the source list
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
	PVRush::PVRawSourceBase_p src = _ext.get_agg().agg_index_to_source(index, &offset);
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
