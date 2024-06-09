//
// MIT License
//
// © ESI Group, 2015
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
//
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
//
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

#include <pvkernel/widgets/PVHelpWidget.h>
#include <qnamespace.h>

#include "pvkernel/widgets/PVPopupWidget.h"
#include "pvkernel/widgets/PVTextPopupWidget.h"

class QWidget;

/*****************************************************************************
 * PVWidgets::PVHelpWidget::PVHelpWidget
 *****************************************************************************/

PVWidgets::PVHelpWidget::PVHelpWidget(QWidget* parent) : PVWidgets::PVTextPopupWidget(parent)
{
}

/*****************************************************************************
 * PVWidgets::PVHelpWidget::is_close_key
 *****************************************************************************/

bool PVWidgets::PVHelpWidget::is_close_key(int key)
{
	if (is_help_key(key)) {
		return true;
	}

	return PVPopupWidget::is_close_key(key);
}

/*****************************************************************************
 * PVWidgets::PVHelpWidget::is_help_key
 *****************************************************************************/

bool PVWidgets::PVHelpWidget::is_help_key(int key)
{
	// TODO : We should use QKeySequence::HelpContents for help
	return ((key == Qt::Key_Question) || (key == Qt::Key_Help) || (key == Qt::Key_F1));
}
