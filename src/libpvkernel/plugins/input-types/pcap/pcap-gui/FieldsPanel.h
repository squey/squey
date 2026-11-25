/* * MIT License
 *
 * © Squey, 2026
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

#ifndef PCAP_GUI_FIELDSPANEL_H
#define PCAP_GUI_FIELDSPANEL_H

#include <libpvpcap/catalog.h>
#include <libpvpcap/profile.h>
#include <libpvpcap/sample.h>

#include <QAbstractItemModel>
#include <QString>
#include <QWidget>

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class QLabel;
class QLineEdit;
class QTimer;
class QTreeView;

namespace PVPcapsicum
{

//! What the first frames of the captures hold, together.
struct sampled_fields {
	uint64_t frames = 0;
	//! The least deep first, then those most frames hold.
	std::vector<pvpcap::capture_sample::protocol> protocols;
	std::unordered_map<std::string, pvpcap::capture_sample::field> fields;

	//! Of the frames holding a field or protocol, from 0 to 1.
	double share(uint64_t holding) const { return frames == 0 ? 0 : double(holding) / frames; }
};

//! Merges what the first frames of each capture hold.
sampled_fields merge_samples(const std::vector<const pvpcap::capture_sample*>& samples);

/**
 * The fields tshark knows, as a tree of the protocols the captures hold and of their fields,
 * or as the fields a search found. Those that are columns are checked.
 */
class FieldTreeModel : public QAbstractItemModel
{
	Q_OBJECT

  public:
	enum column { NAME, DESCRIPTION, SHARE, EXAMPLE, COLUMN_COUNT };
	//! Of a row: its share of the sampled frames, from 0 to 1, or -1 if none was sampled.
	static constexpr int SHARE_ROLE = Qt::UserRole;
	//! Of a row of a field: its filter name.
	static constexpr int FIELD_ROLE = Qt::UserRole + 1;
	static constexpr char MIME_TYPE[] = "application/x-squey-pcap-fields";

	explicit FieldTreeModel(QObject* parent = nullptr);

	void set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog);
	void set_samples(sampled_fields samples);
	void set_columns(const std::vector<pvpcap::profile_field>& columns);
	/**
	 * Shows the fields of the search rather than the tree, the best first.
	 *
	 * @return how many fields the search found, of which `most` at most are shown
	 */
	size_t set_search(const QString& search, size_t most);

	const sampled_fields& samples() const { return _samples; }

	QModelIndex index(int row, int column, const QModelIndex& parent = {}) const override;
	QModelIndex parent(const QModelIndex& index) const override;
	int rowCount(const QModelIndex& parent = {}) const override;
	int columnCount(const QModelIndex& parent = {}) const override;
	QVariant data(const QModelIndex& index, int role) const override;
	bool setData(const QModelIndex& index, const QVariant& value, int role) override;
	QVariant headerData(int section, Qt::Orientation orientation, int role) const override;
	Qt::ItemFlags flags(const QModelIndex& index) const override;
	QStringList mimeTypes() const override;
	QMimeData* mimeData(const QModelIndexList& indexes) const override;

  Q_SIGNALS:
	//! A field was checked, or unchecked.
	void toggled(const pvpcap::profile_field& field, bool checked);

  private:
	struct node {
		enum class kind { protocol, field, others } type = kind::field;
		//! Of the protocol or field in the catalog, or how many fields `others` holds.
		uint32_t index = 0;
		int parent = -1;
		int row = 0;
		std::vector<int> children;
		uint64_t frames = 0;
		int columns = 0; //!< of a protocol: how many of its fields are columns
	};

	void rebuild();
	//! @param parent -1 for a row at the top
	int add(node n, int parent);
	void add_fields(int protocol, const std::vector<uint32_t>& fields, bool split);
	bool is_column(uint32_t field) const;
	bool importable(uint32_t field) const;
	QString tooltip(const node& n) const;

	std::shared_ptr<const pvpcap::field_catalog> _catalog;
	sampled_fields _samples;
	std::vector<pvpcap::profile_field> _columns;
	std::unordered_set<std::string> _column_names;
	QString _search;
	size_t _most = 0;
	size_t _found = 0;
	std::vector<node> _nodes;
	std::vector<int> _roots;
};

/**
 * Where the columns are chosen: the protocols the captures hold and their fields, or the
 * fields a search finds among those tshark knows.
 */
class FieldsPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit FieldsPanel(QWidget* parent = nullptr);

	//! Null while it is loaded, with why if it could not be.
	void set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog,
	                 const QString& problem = {});
	void set_samples(sampled_fields samples, bool reading);
	void set_columns(const std::vector<pvpcap::profile_field>& columns);
	void focus_search();
	//! Finds the field, and selects it.
	void reveal(const std::string& name);

	const sampled_fields& samples() const { return _model->samples(); }

  Q_SIGNALS:
	//! A field is to be a column, or no longer to be one.
	void toggled(const pvpcap::profile_field& field, bool column);

  protected:
	bool eventFilter(QObject* watched, QEvent* event) override;

  private:
	void search();
	void update_status();
	//! Checks the field of the row, or unchecks it.
	void toggle(const QModelIndex& index);

	FieldTreeModel* _model;
	QLineEdit* _search;
	QTreeView* _view;
	QLabel* _status;
	QTimer* _searching;
	QString _catalog_problem;
	bool _catalog_loaded = false;
	bool _reading = false;
	size_t _found = 0;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_FIELDSPANEL_H
