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

#ifndef PCAP_GUI_FILTERPANEL_H
#define PCAP_GUI_FILTERPANEL_H

#include <libpvpcap/profile.h>
#include <libpvpcap/ws.h>

#include <QLineEdit>
#include <QWidget>

#include <optional>

class QCompleter;
class QLabel;
class QStringListModel;

namespace PVPcapsicum
{

/**
 * A display filter of Wireshark, whose names of fields and protocols are completed as they
 * are typed. Enter is kept from the dialog, which would import.
 */
class FilterEdit : public QLineEdit
{
	Q_OBJECT

  public:
	explicit FilterEdit(QWidget* parent = nullptr);

	//! Of the fields and the protocols, sorted whatever the case of their letters.
	void set_names(const QStringList& names);
	//! Draws the filter as wrong, or right if `wrong` is false.
	void set_wrong(bool wrong);

  protected:
	void keyPressEvent(QKeyEvent* event) override;

  private:
	void complete();
	void insert_completion(const QString& completion);
	//! Of the name the cursor ends.
	int name_start() const;

	QCompleter* _completer;
	QStringListModel* _names;
};

/**
 * Which frames are imported: the display filter of the profile, which tshark checks as it is
 * typed. Giving it the focus gives it to the filter.
 */
class FilterPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit FilterPanel(QWidget* parent = nullptr);

	//! Of the fields and the protocols, sorted whatever the case of their letters.
	void set_names(const QStringList& names);
	//! Shows the filter of the profile, without telling it was edited.
	void set_profile(const pvpcap::profile& p);
	//! What tshark said of the filter shown, or nothing while it checks it.
	void set_check(const std::optional<pvpcap::filter_check>& check);
	//! Such as how many of the frames of the preview the filter keeps.
	void set_note(const QString& note);

  Q_SIGNALS:
	void edited(const pvpcap::profile& p);

  private:
	void edit();
	//! In the colors of the theme, which may change.
	void update_links();

	FilterEdit* _edit;
	QLabel* _syntax;
	QLabel* _status;
	QString _note;
	//! What set_check() was told last.
	std::optional<pvpcap::filter_check> _shown_check;
	pvpcap::filter_check _check;
	pvpcap::profile _profile;
	bool _showing = false;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_FILTERPANEL_H
