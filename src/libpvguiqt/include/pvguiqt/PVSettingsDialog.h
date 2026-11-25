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

#ifndef PVGUIQT_PVSETTINGSDIALOG_H
#define PVGUIQT_PVSETTINGSDIALOG_H

#include <pvguiqt/export.h>

#include <QDialog>
#include <QHash>
#include <QJsonObject>
#include <QString>

#include <vector>

class QDialogButtonBox;
class QLabel;
class QLineEdit;
class QListWidget;
class QScrollArea;
class QStackedWidget;
class QTextEdit;
class QTimer;
class QToolButton;
class QTreeWidget;

namespace PVGuiQt
{

namespace __impl
{
class setting_row;
} // namespace __impl

/**
 * Searches and changes the settings of PVCore::PVSettings, one by one, or as
 * the JSON of settings.json beside the default settings.
 *
 * Changes are made to a draft of the settings of the user: OK and Apply save it,
 * Cancel throws it away.
 */
class PVGUIQT_EXPORT PVSettingsDialog : public QDialog
{
	Q_OBJECT

  public:
	explicit PVSettingsDialog(QWidget* parent = nullptr);
	~PVSettingsDialog() override;

  public:
	/**
	 * Lists the settings that match every word of \p filter, in their key, title,
	 * description, section or values, regardless of case. The word "@modified"
	 * keeps the settings the draft sets.
	 */
	void set_filter(QString const& filter);

	/** Edits the draft as the text of settings.json rather than setting by setting. */
	void set_json_mode(bool json);
	bool json_mode() const;

	/** The settings of the user as OK and Apply save them, unknown keys included. */
	QJsonObject draft() const;

	/**
	 * Saves the draft.
	 *
	 * \return false when the JSON being edited does not parse, or when the
	 *         settings cannot be written: the dialog then says why.
	 */
	bool apply();

	void accept() override;

  private:
	void build_form();
	void build_json_page();
	/** The default settings, as settings.json would write them, with their descriptions. */
	QString defaults_document() const;
	void refresh_rows();
	void update_visibility();
	void set_draft_value(QString const& key, QVariant const& value);
	void reset_draft_value(QString const& key);
	void restore_listed_defaults();
	void check_json();
	void show_help_at_cursor(QTextEdit* editor);
	/** Takes the text of settings.json being edited as the draft, if it parses. */
	bool take_json();
	void show_notice(QString const& text);

  private:
	QJsonObject _draft;
	QStringList _filter_words;
	bool _modified_only = false;
	/** The last save changed a setting that takes a restart. */
	bool _restart_needed = false;

	/** The sections that hold settings or sections, in the order of the form. */
	std::vector<QString> _display_sections;
	/** "Plugins › Layer filters", by key of section. */
	QHash<QString, QString> _full_titles;
	QHash<QString, QString> _descriptions;

	QLineEdit* _search;
	QLabel* _count;
	QToolButton* _json_toggle;
	QStackedWidget* _pages;
	QTreeWidget* _sections;
	QScrollArea* _form;
	QLabel* _no_match;
	std::vector<__impl::setting_row*> _rows;
	/** The header of each section of the form, by key of section. */
	QHash<QString, QWidget*> _headers;

	QTextEdit* _defaults_json;
	QTextEdit* _user_json;
	QListWidget* _problems;
	QLabel* _json_help;
	QTimer* _check_timer;

	QLabel* _notice;
	QDialogButtonBox* _buttons;
};

} // namespace PVGuiQt

#endif // PVGUIQT_PVSETTINGSDIALOG_H
