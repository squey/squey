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

#ifndef PCAP_GUI_COLUMNSPANEL_H
#define PCAP_GUI_COLUMNSPANEL_H

#include "FieldsPanel.h"

#include <libpvpcap/catalog.h>
#include <libpvpcap/profile.h>

#include <QStringList>
#include <QWidget>

#include <map>
#include <memory>
#include <string>
#include <vector>

class QLabel;
class QListWidget;
class QToolButton;

namespace PVPcapsicum
{

/**
 * The columns of the import, in their order: moved by dragging them, or with Alt+Up and
 * Alt+Down, and removed with the Delete key. Fields dropped on it become columns where they
 * are dropped.
 */
class ColumnsPanel : public QWidget
{
	Q_OBJECT

  public:
	explicit ColumnsPanel(QWidget* parent = nullptr);

	void set_catalog(std::shared_ptr<const pvpcap::field_catalog> catalog);
	void set_samples(const sampled_fields& samples);
	void set_columns(const std::vector<pvpcap::profile_field>& columns);
	//! Of each column, the share of the rows of the preview with a value, from 0 to 1.
	void set_filled(const std::map<std::string, double>& filled);

  Q_SIGNALS:
	//! The columns were moved or removed.
	void edited(const std::vector<pvpcap::profile_field>& columns);
	//! Fields were dropped on the list, to be inserted at `row`.
	void dropped(const QStringList& fields, int row);
	//! The field of a column is to be shown among the others.
	void reveal(const std::string& field);

  private:
	void refresh();
	void emit_edited();
	void move_selected(int by);
	void remove_selected();
	void remove_row(int row);
	void update_header();

	QListWidget* _list;
	QLabel* _count;
	QToolButton* _up;
	QToolButton* _down;
	QToolButton* _remove;
	std::shared_ptr<const pvpcap::field_catalog> _catalog;
	sampled_fields _samples;
	std::vector<pvpcap::profile_field> _columns;
	std::map<std::string, double> _filled;
};

} // namespace PVPcapsicum

#endif // PCAP_GUI_COLUMNSPANEL_H
