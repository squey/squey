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
