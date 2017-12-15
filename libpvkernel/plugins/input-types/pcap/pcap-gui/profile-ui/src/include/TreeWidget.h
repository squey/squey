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
	ExtractProtocolsWorker(pvpcap::PcapTreeModel* m, const QString& p)
	    : _tree_model(m), _pcap_path(p){};

  Q_SIGNALS:
	void finished();

  public Q_SLOTS:
	void load()
	{
		_tree_model->load(_pcap_path, _canceled);
		Q_EMIT finished();
	}
	void cancel() { _canceled = true; }

  private:
	pvpcap::PcapTreeModel* _tree_model;
	const QString& _pcap_path;
	bool _canceled = false;
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
