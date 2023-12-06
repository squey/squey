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

#include <PVXmlParamColorDialog.h>

/******************************************************************************
 *
 * App::PVXmlParamColorDialog::PVXmlParamColorDialog
 *
 *****************************************************************************/
App::PVXmlParamColorDialog::PVXmlParamColorDialog(QString name,
                                                          QString p_color,
                                                          QWidget* p_parent)
    : QPushButton(p_color, p_parent)
{
	setObjectName(name);
	setColor(p_color);
	setText(p_color);
	parent = p_parent;
	connect(this, &QAbstractButton::clicked, this, &PVXmlParamColorDialog::chooseColor);
}

/******************************************************************************
 *
 * App::PVXmlParamColorDialog::~PVXmlParamColorDialog
 *
 *****************************************************************************/
App::PVXmlParamColorDialog::~PVXmlParamColorDialog()
{
	disconnect(this, &QAbstractButton::clicked, this, &PVXmlParamColorDialog::chooseColor);
}

/******************************************************************************
 *
 * App::PVXmlParamColorDialog::chooseColor
 *
 *****************************************************************************/
void App::PVXmlParamColorDialog::chooseColor()
{
	// qDebug()<<"PVXmlParamColorDialog::chooseColor()";

	QColorDialog cd(parent);
	QColor initialColor(color);
	QColor colorChoosed = cd.getColor(initialColor, parent);
	if (colorChoosed.isValid()) {
		setColor(colorChoosed.name());
		Q_EMIT changed();
	}
}

/******************************************************************************
 *
 * App::PVXmlParamColorDialog::setColor
 *
 *****************************************************************************/
void App::PVXmlParamColorDialog::setColor(QString newColor)
{
	color = newColor;
	setText(newColor);
	QString css = QString("background-color:%1;").arg(color);
	setStyleSheet(css);
}

/******************************************************************************
 *
 * App::PVXmlParamColorDialog::getColor
 *
 *****************************************************************************/
QString App::PVXmlParamColorDialog::getColor()
{
	return color;
}
