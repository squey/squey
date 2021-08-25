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

#ifndef PROGRESSWIDGET_H
#define PROGRESSWIDGET_H

#include <QWidget>
#include <QThread>

#include "libpvpcap/pcap_splitter.h"

#include <pvlogger.h>

namespace Ui
{
class ProgressWidget;
}

Q_DECLARE_METATYPE(size_t)

class PcapPreprocessingThread : public QThread
{
	Q_OBJECT;

  public:
	PcapPreprocessingThread(const QStringList& filenames,
	                        const std::vector<std::string>& tshark_cmd)
	    : _filenames(filenames), _tshark_cmd(tshark_cmd)
	{
	}

  public:
	pvpcap::splitted_files_t csv_paths() const { return _csv_paths; }

  protected:
	void run();

  public Q_SLOTS:
	void cancel() { _canceled = true; }

  Q_SIGNALS:
	void split_datasize(size_t datasize);
	void split_progress(size_t datasize);
	void extract_packetscount(size_t packets_count);
	void extract_progress(size_t packets_count);
	void finished();

  private:
	const QStringList& _filenames;
	pvpcap::splitted_files_t _csv_paths;
	std::vector<std::string> _tshark_cmd;
	bool _canceled = false;
};

/**
 * It is the UI for Progressing job running process.
 */
class ProgressWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit ProgressWidget(const QStringList& filenames,
	                        const std::vector<std::string>& tshark_cmd,
	                        QWidget* parent = 0);
	~ProgressWidget();

  public:
	void run();
	pvpcap::splitted_files_t csv_paths() const { return _thread.csv_paths(); }
	bool is_canceled() const { return _canceled; }

  private Q_SLOTS:
	void set_split_progress_maximum(size_t max);
	void set_split_progress_value(size_t value);
	void set_extract_progress_value(size_t value);
	void set_extract_progress_maximum(size_t max);

  Q_SIGNALS:
	void closed();

  private:
	Ui::ProgressWidget* _ui; //!< The ui generated interface.
	PcapPreprocessingThread _thread;
	bool _canceled = false;
};

#endif // PROGRESSWIDGET_H
