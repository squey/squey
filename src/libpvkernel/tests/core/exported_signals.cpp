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

// Libraries and executables other than pvkernel connect these signals by address. Qt finds a
// signal by comparing that address with the one the meta-object of pvkernel holds, and MinGW
// makes the two equal only for an exported class: for another one, the address taken outside
// is that of an import thunk, and the connection silently fails under Windows.

#include <pvkernel/core/PVProgressBox.h>
#include <pvkernel/core/squey_assert.h>
#include <pvkernel/rush/PVXmlTreeNodeDom.h>
#include <pvkernel/widgets/PVArgumentListWidget.h>
#include <pvkernel/widgets/PVLongLongSpinBox.h>
#include <pvkernel/widgets/PVPresetsWidget.h>

#include <QMetaMethod>

int main()
{
	PV_ASSERT_VALID(QMetaMethod::fromSignal(&PVCore::PVProgressBox::cancel_asked_sig).isValid());
	PV_ASSERT_VALID(QMetaMethod::fromSignal(&PVRush::PVXmlTreeNodeDom::data_changed).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVArgumentListWidget::args_changed_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVPresetsWidget::btn_load_clicked_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVPresetsWidget::btn_new_clicked_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVPresetsWidget::btn_save_clicked_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVPresetsWidget::btn_remove_clicked_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(&PVWidgets::PVPresetsWidget::preset_renamed_Signal).isValid());
	PV_ASSERT_VALID(
	    QMetaMethod::fromSignal(qOverload<qlonglong>(&PVWidgets::PVLongLongSpinBox::valueChanged))
	        .isValid());

	return 0;
}
