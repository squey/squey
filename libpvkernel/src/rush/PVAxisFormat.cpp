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

#include <pvkernel/rush/PVAxisFormat.h> // for PVAxisFormat

#include <pvkernel/core/PVColor.h> // for PVColor

#include <pvbase/types.h> // for PVCol

/******************************************************************************
 *
 * PVAxisFormat
 *
 *****************************************************************************/

PVRush::PVAxisFormat::PVAxisFormat(PVCol index) : index(index)
{
}

void PVRush::PVAxisFormat::set_color(QString str)
{
	color.fromQColor(QColor(str));
}

void PVRush::PVAxisFormat::set_color(PVCore::PVColor color_)
{
	color = color_;
}

void PVRush::PVAxisFormat::set_mapping(QString str)
{
	mapping = str;
}

void PVRush::PVAxisFormat::set_name(const QString& str)
{
	name = str;
}

void PVRush::PVAxisFormat::set_scaling(QString str)
{
	scaling = str;
}

void PVRush::PVAxisFormat::set_titlecolor(QString str)
{
	titlecolor.fromQColor(QColor(str));
}

void PVRush::PVAxisFormat::set_titlecolor(PVCore::PVColor color_)
{
	titlecolor = color_;
}

void PVRush::PVAxisFormat::set_type(QString str)
{
	type = str;
}

void PVRush::PVAxisFormat::set_type_format(QString str)
{
	type_format = str;
}
