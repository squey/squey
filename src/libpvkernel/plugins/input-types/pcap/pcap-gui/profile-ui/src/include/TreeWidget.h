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

#ifndef TREEWIDGET_H
#define TREEWIDGET_H

#include <QWidget>
#include <QThread>

#include <profile-ui/models/include/PcapTreeModel.h>

namespace Ui
{
class TreeWidget;
}

class QProgressDialog;
class QThread;

class ExtractProtocolsWorker : public QObject
{
	Q_OBJECT

  public:
	explicit ExtractProtocolsWorker(const QString& p) : _pcap_path(p) {}

	//! What was read, once finished() was emitted.
	const rapidjson::Document& protocols() const { return _protocols; }
	//! Why nothing was read, once finished() was emitted.
	const QString& error() const { return _error; }
	bool canceled() const { return _canceled; }

  Q_SIGNALS:
	void finished();

  public Q_SLOTS:
	void load()
	{
		try {
			_protocols = pvpcap::PcapTreeModel::read_protocols(_pcap_path, _canceled);
		} catch (const std::exception& e) {
			_error = QString::fromStdString(e.what());
		}
		Q_EMIT finished();
	}

	//! Called from another thread than the one load() runs in.
	void cancel() { _canceled = true; }

  private:
	QString _pcap_path;
	rapidjson::Document _protocols;
	QString _error;
	std::atomic<bool> _canceled = false;
};

/**
 * It is the UI for monitoring job running process.
 */
class TreeWidget : public QWidget
{
	Q_OBJECT

  public:
	explicit TreeWidget(rapidjson::Document* json_data, QWidget* parent = 0);
	~TreeWidget();

  Q_SIGNALS:
	/**
	 * Signal emitted when the Tree information is updated.
	 */
	void update_tree_data();

	/**
	 * Emit this signal to say that selection have changed.
	 */
	void propagate_selection();

  private Q_SLOTS:
	void on_select_button_clicked();
	void update_field_model(const QModelIndex& index);
	void update_field_model_with_selected_field(const QModelIndex& index);
	/**
	 * As a field's selection changed, update the TreeModel.
	 */
	void update_selection();

  public:
	void update_model();

  private:
	Ui::TreeWidget* _ui;                //!< The ui generated interface.
	pvpcap::PcapTreeModel* _tree_model; //!< Model to display the pcap Tree
	QModelIndex _selected_protocol;     //!< Currently selected protocol
};

#endif // TREEWIDGET_H
