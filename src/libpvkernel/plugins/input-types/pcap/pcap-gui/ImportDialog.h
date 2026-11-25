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

#ifndef PCAP_GUI_IMPORTDIALOG_H
#define PCAP_GUI_IMPORTDIALOG_H

#include <libpvpcap/catalog.h>
#include <libpvpcap/profile.h>
#include <libpvpcap/ws.h>

#include <QDialog>
#include <QDomDocument>
#include <QStringList>

#include <memory>
#include <optional>
#include <string>

class QLabel;
class QPushButton;
class QSplitter;
class QTabWidget;
class QTimer;

namespace PVPcapsicum
{

class BackgroundJob;
class CapturesPanel;
class ColumnsPanel;
class FieldsPanel;
class FilterPanel;
class OptionsPanel;
class PreviewPanel;
class ProfileEditor;
class ProfilesPanel;

/**
 * Where captures are imported from: the captures, the profile saying what to import of
 * them and how, the fields to choose its columns from, and a preview of what the import
 * makes of the first frames.
 *
 * What is changed of the profiles is kept until it is saved or reverted, as is what the
 * dialog looked like: the next one opens as this one closes.
 */
class ImportDialog : public QDialog
{
	Q_OBJECT

  public:
	explicit ImportDialog(const QStringList& captures, QWidget* parent = nullptr);
	~ImportDialog() override;

	//! In the order their frames are numbered in.
	QStringList captures() const;
	//! Adds the files, and the captures in the directories, which are not there yet.
	void add_captures(const QStringList& paths);
	//! What to import, once the dialog was accepted.
	const pvpcap::profile& profile() const { return _accepted; }
	//! The format of what is imported, once the dialog was accepted.
	QDomDocument format() const;

  public Q_SLOTS:
	//! Imports, unless what to import is not right yet: says then what is wrong.
	void accept() override;
	void reject() override;

  protected:
	void dragEnterEvent(QDragEnterEvent* event) override;
	void dropEvent(QDropEvent* event) override;

  private:
	void build();
	void connect_parts();
	void load_catalog();
	void show_profile();
	void show_samples();
	void check_filter();
	void update_header();
	void update_summary();
	void set_columns(std::vector<pvpcap::profile_field> columns);
	void toggle_field(const pvpcap::profile_field& field, bool column);
	void insert_fields(const QStringList& names, int row);
	void save();
	void save_as();
	void remember();
	//! Why the import can't be done yet, empty if it can.
	QString problem() const;
	//! The columns and the fields of the filter making tshark read the captures twice.
	QStringList two_pass_fields() const;

	ProfileEditor* _editor;
	CapturesPanel* _captures;
	ProfilesPanel* _profiles;
	FieldsPanel* _fields;
	ColumnsPanel* _columns;
	OptionsPanel* _options;
	QTabWidget* _tabs;
	FilterPanel* _filter;
	PreviewPanel* _preview;
	QSplitter* _main;
	QSplitter* _vertical;
	QSplitter* _choice;
	QLabel* _profile_name;
	QLabel* _profile_state;
	QPushButton* _revert;
	QPushButton* _save;
	QPushButton* _save_as;
	QLabel* _summary;
	QLabel* _problem;
	QPushButton* _import;

	//! What keeps the catalog of the dialogs open.
	std::shared_ptr<const void> _loaded;
	std::shared_ptr<const pvpcap::field_catalog> _catalog;
	BackgroundJob* _catalog_job;
	BackgroundJob* _filter_job;
	QTimer* _filter_waiting;
	//! What tshark said of the filter it checked last.
	std::optional<pvpcap::filter_check> _filter_check;
	std::string _checked_filter;
	//! The filter tshark is to check, or checks.
	std::optional<std::string> _pending_filter;
	pvpcap::profile _accepted;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_IMPORTDIALOG_H
