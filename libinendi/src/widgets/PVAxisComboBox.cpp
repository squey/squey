/**
 * @file
 *
 * @copyright (C) ESI Group INENDI April 2019
 */

#include <inendi/widgets/PVAxisComboBox.h>
#include <pvkernel/rush/PVFormat.h>

#include <QMouseEvent>
#include <QApplication>
#include <QDrag>
#include <QDragEnterEvent>
#include <QDragLeaveEvent>
#include <QDragMoveEvent>
#include <QDropEvent>
#include <QMimeData>

PVWidgets::PVAxisComboBox::PVAxisComboBox(Inendi::PVAxesCombination const& axes_comb,
                                          AxesShown shown,
                                          axes_filter_t axes_filter,
                                          QWidget* parent)
    : QComboBox(parent), _axes_comb(axes_comb), _axes_shown(shown), _axes_filter(axes_filter)
{
	refresh_axes();
	connect(this, QOverload<int>::of(&QComboBox::currentIndexChanged), [this](int index) {
		if (index < 0) {
			return current_axis_changed(PVCol(), PVCombCol());
		}
		if (_axes_shown == AxesShown::CombinationAxes or
		    (_axes_shown == AxesShown::BothOriginalCombinationAxes and
		     index < count() - _axes_comb.get_nraw_axes_count())) {
			return current_axis_changed(current_axis(), PVCombCol(index));
		}
		return current_axis_changed(current_axis(), PVCombCol());
	});
	setAcceptDrops(true);
}

void PVWidgets::PVAxisComboBox::set_current_axis(PVCombCol axis)
{
	if (axis == PVCombCol()) {
		setCurrentIndex(-1);
	} else if (_axes_shown & AxesShown::CombinationAxes) {
		setCurrentIndex(findData(QVariant::fromValue(axis)));
	} else {
		setCurrentIndex(findData(QVariant::fromValue(_axes_comb.get_nraw_axis(axis))));
	}
}

void PVWidgets::PVAxisComboBox::set_current_axis(PVCol axis)
{
	if (axis == PVCol()) {
		setCurrentIndex(-1);
	} else if (_axes_shown == AxesShown::OriginalAxes) {
		setCurrentIndex(findData(QVariant::fromValue(axis)));
	} else if (_axes_shown == AxesShown::BothOriginalCombinationAxes) {
		setCurrentIndex(findData(QVariant::fromValue(axis)));
	} else {
		setCurrentIndex(findData(QVariant::fromValue(_axes_comb.get_first_comb_col(axis))));
	}
}

PVCol PVWidgets::PVAxisComboBox::current_axis() const
{
	if (currentIndex() == -1) {
		return PVCol();
	}
	return currentData().value<PVCol>();
}

void PVWidgets::PVAxisComboBox::refresh_axes()
{
	clear();
	if (_axes_shown & AxesShown::CombinationAxes) {
		for (auto i = PVCombCol(0); i < _axes_comb.get_axes_count(); ++i) {
			if (_axes_filter(PVCol(), i)) {
				addItem(_axes_comb.get_axis(i).get_name(),
				        QVariant::fromValue(_axes_comb.get_nraw_axis(i)));
			}
		}
	}
	if (_axes_shown == AxesShown::BothOriginalCombinationAxes) {
		insertSeparator(count());
	}
	if (_axes_shown & AxesShown::OriginalAxes) {
		for (auto i = PVCol(0); i < _axes_comb.get_nraw_axes_count(); ++i) {
			if (_axes_filter(i, PVCombCol())) {
				addItem(_axes_comb.get_axis(i).get_name(), QVariant::fromValue(i));
			}
		}
	}
}

void PVWidgets::PVAxisComboBox::mousePressEvent(QMouseEvent* event)
{
	if (event->button() == Qt::LeftButton) {
		_drag_start_position = event->pos();
	}
	QComboBox::mousePressEvent(event);
}

void PVWidgets::PVAxisComboBox::mouseMoveEvent(QMouseEvent* event)
{
	if (!(event->buttons() & Qt::LeftButton))
		return;
	if ((event->pos() - _drag_start_position).manhattanLength() < QApplication::startDragDistance())
		return;

	QDrag* drag = new QDrag(this);
	QMimeData* mimeData = new QMimeData;

	mimeData->setData(MIME_TYPE_PVCOL,
	                  QByteArray::fromRawData(
	                      reinterpret_cast<char const*>(&current_axis().value()), sizeof(PVCol)));
	drag->setMimeData(mimeData);

	Qt::DropAction dropAction = drag->exec(Qt::CopyAction);
}

void PVWidgets::PVAxisComboBox::dragEnterEvent(QDragEnterEvent* event)
{
	if (event->mimeData()->hasFormat(MIME_TYPE_PVCOL)) {
		event->acceptProposedAction();
	}
}

void PVWidgets::PVAxisComboBox::dropEvent(QDropEvent* event)
{
	PVCol axis_col(
	    *reinterpret_cast<PVCol::value_type*>(event->mimeData()->data(MIME_TYPE_PVCOL).data()));

	set_current_axis(axis_col);

	event->acceptProposedAction();
}