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

#include "include/ProgressWidget.h"

#include "libpvpcap/shell.h"

#include <pvkernel/rush/PVNrawCacheManager.h>

#include <thread>

#include <pvlogger.h>

#include <pvhwloc.h>

#include "ui_ProgressWidget.h"

static constexpr const size_t MEGABYTE = 1 * 1024 * 1024;

ProgressWidget::ProgressWidget(const QStringList& filenames,
                               const std::vector<std::string>& tshark_cmd,
                               QWidget* parent)
    : QWidget(parent), _ui(new Ui::ProgressWidget), _thread(filenames, tshark_cmd)
{
	_ui->setupUi(this);

	_ui->_split_progressbar->setTextVisible(true);
	_ui->_extract_progressbar->setTextVisible(true);

	qRegisterMetaType<size_t>("size_t");
	connect(&_thread, &PcapPreprocessingThread::split_datasize, this,
	        &ProgressWidget::set_split_progress_maximum, Qt::QueuedConnection);
	connect(&_thread, &PcapPreprocessingThread::split_progress, this,
	        &ProgressWidget::set_split_progress_value, Qt::QueuedConnection);
	connect(&_thread, &PcapPreprocessingThread::extract_packetscount, this,
	        &ProgressWidget::set_extract_progress_maximum, Qt::QueuedConnection);
	connect(&_thread, &PcapPreprocessingThread::extract_progress, this,
	        &ProgressWidget::set_extract_progress_value, Qt::QueuedConnection);
	connect(&_thread, &PcapPreprocessingThread::finished, [&]() {
		close();
		Q_EMIT closed();
	});
	connect(_ui->_cancel_button, &QPushButton::clicked, [&]() {
		_canceled = true;
		_thread.cancel();
	});
}

ProgressWidget::~ProgressWidget()
{
	_thread.wait();
	delete _ui;
}

void ProgressWidget::run()
{
	_thread.start();
}

void PcapPreprocessingThread::run()
{
	std::vector<std::string> pcap_files(_filenames.size());
	std::transform(_filenames.begin(), _filenames.end(), pcap_files.begin(),
	               [](QString v) { return v.toStdString(); });

	_csv_paths = pvpcap::extract_csv(
	    pvpcap::split_pcaps(pcap_files, PVRush::PVNrawCacheManager::nraw_dir().toStdString(),
	                        true /* preserve flows */, _canceled,
	                        [&](size_t datasize) { Q_EMIT split_datasize(datasize); },
	                        [&](size_t progress) { Q_EMIT split_progress(progress); }),
	    _tshark_cmd, _canceled,
	    [&](size_t packets_count) { Q_EMIT extract_packetscount(packets_count); },
	    [&](size_t progress) { Q_EMIT extract_progress(progress); });

	Q_EMIT finished();
}

void ProgressWidget::set_split_progress_maximum(size_t max)
{
	_ui->_split_progressbar->setMaximum(max / MEGABYTE);
}

void ProgressWidget::set_split_progress_value(size_t value)
{
	_ui->_split_progressbar->setValue(value / MEGABYTE);
	_ui->_split_progressbar->setFormat(
	    QString("%L1").arg(_ui->_split_progressbar->value()) + " / " +
	    QString("%L1").arg(_ui->_split_progressbar->maximum()) + " MB" + " (" +
	    QString::number((size_t)((double)_ui->_split_progressbar->value() /
	                             _ui->_split_progressbar->maximum() * 100)) +
	    "%)");
}

void ProgressWidget::set_extract_progress_maximum(size_t max)
{
	_ui->_extract_progressbar->setMaximum(max);
	_ui->_extract_progressbar->setFormat(QString("0 / ") +
	                                     QString("%L1").arg(_ui->_extract_progressbar->maximum()) +
	                                     " packets (0%)");
}

void ProgressWidget::set_extract_progress_value(size_t value)
{
	_ui->_extract_progressbar->setValue(value);
	_ui->_extract_progressbar->setFormat(
	    QString("%L1").arg(_ui->_extract_progressbar->value()) + " / " +
	    QString("%L1").arg(_ui->_extract_progressbar->maximum()) + " packets" + " (" +
	    QString::number((size_t)((double)_ui->_extract_progressbar->value() /
	                             _ui->_extract_progressbar->maximum() * 100)) +
	    "%)");
}
