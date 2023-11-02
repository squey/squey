/* * MIT License
 *
 * © Florent Chapelle, 2023
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef PVDISPLAYS_PVDISPLAYVIEWGROUPBY_H
#define PVDISPLAYS_PVDISPLAYVIEWGROUPBY_H

#include <pvkernel/core/PVRegistrableClass.h>
#include <pvkernel/widgets/PVModdedIcon.h>
#include <pvdisplays/PVDisplayIf.h>

#include <squey/PVSelection.h>

namespace PVGuiQt
{
	class PVGroupByStringsDlg;
}

namespace PVDisplays
{

class PVDisplayViewGroupBy : public PVDisplayViewIf
{
  public:
	using PVDisplayViewIf::PVDisplayViewIf;

  public:
	QWidget*
	create_widget(Squey::PVView* view, QWidget* parent, Params const& data = {}) const override;

	void add_to_axis_menu(QMenu& menu, PVCol axis, PVCombCol axis_comb,
						  Squey::PVView* view, PVDisplaysContainer* container) override;
  protected:
	virtual bool is_groupable_by(QString const& type) = 0;
	virtual auto show_group_by(Squey::PVView& view,
	                           PVCol col1,
	                           PVCol col2,
	                           Squey::PVSelection const& sel,
	                           QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* = 0;
};

class PVDisplayViewCountBy : public PVDisplayViewGroupBy
{
  public:
	PVDisplayViewCountBy(): PVDisplayViewGroupBy(ShowInCtxtMenu,
		QObject::tr("Count by"), PVModdedIcon("count-by"), QObject::tr("Count by"), Qt::LeftDockWidgetArea)
	{}

	bool is_groupable_by(QString const&) override { return true; }
	
	auto show_group_by(Squey::PVView& view,
	                   PVCol col1,
	                   PVCol col2,
	                   Squey::PVSelection const& sel,
	                   QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* override;

	CLASS_REGISTRABLE(PVDisplayViewCountBy)
};

static inline const QStringList summable_types = {
	"number_int64",  "number_uint64", "number_int32", "number_uint32",
	"number_uint16", "number_int16",  "number_uint8", "number_int8",
	"number_float",  "number_double", "duration"
};

class PVDisplayViewSumBy : public PVDisplayViewGroupBy
{
  public:
	PVDisplayViewSumBy(): PVDisplayViewGroupBy(ShowInCtxtMenu,
		QObject::tr("Sum by"), PVModdedIcon("sigma"), QObject::tr("Sum by"), Qt::LeftDockWidgetArea)
	{}

	bool is_groupable_by(QString const& axis_type) override { return summable_types.contains(axis_type); }

	auto show_group_by(Squey::PVView& view, PVCol col1, PVCol col2,
					   Squey::PVSelection const& sel, QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* override;

	CLASS_REGISTRABLE(PVDisplayViewSumBy)
};

class PVDisplayViewMinBy : public PVDisplayViewGroupBy
{
  public:
	PVDisplayViewMinBy(): PVDisplayViewGroupBy(ShowInCtxtMenu,
		QObject::tr("Min by"), PVModdedIcon("arrow-down-to-line"), QObject::tr("Min by"), Qt::LeftDockWidgetArea)
	{}

	bool is_groupable_by(QString const& axis_type) override { return summable_types.contains(axis_type); }

	auto show_group_by(Squey::PVView& view, PVCol col1, PVCol col2,
					   Squey::PVSelection const& sel, QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* override;

	CLASS_REGISTRABLE(PVDisplayViewMinBy)
};

class PVDisplayViewMaxBy : public PVDisplayViewGroupBy
{
  public:
	PVDisplayViewMaxBy(): PVDisplayViewGroupBy(ShowInCtxtMenu,
		QObject::tr("Max by"), PVModdedIcon("arrow-up-to-line"), QObject::tr("Max by"), Qt::LeftDockWidgetArea)
	{}

	bool is_groupable_by(QString const& axis_type) override { return summable_types.contains(axis_type); }

	auto show_group_by(Squey::PVView& view, PVCol col1, PVCol col2,
					   Squey::PVSelection const& sel, QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* override;

	CLASS_REGISTRABLE(PVDisplayViewMaxBy)
};

class PVDisplayViewAverageBy : public PVDisplayViewGroupBy
{
  public:
	PVDisplayViewAverageBy(): PVDisplayViewGroupBy(ShowInCtxtMenu,
		QObject::tr("Average by"), PVModdedIcon("average-by"), QObject::tr("Average by"), Qt::LeftDockWidgetArea)
	{}

	bool is_groupable_by(QString const& axis_type) override { return summable_types.contains(axis_type); }

	auto show_group_by(Squey::PVView& view, PVCol col1, PVCol col2,
					   Squey::PVSelection const& sel, QWidget* parent) const -> PVGuiQt::PVGroupByStringsDlg* override;

	CLASS_REGISTRABLE(PVDisplayViewAverageBy)
};

} // namespace PVDisplays

#endif
