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

#include <inendi/widgets/PVAxisTypeWidget.h>

PVWidgets::PVAxisTypeWidget::PVAxisTypeWidget(QWidget* parent) : PVComboBox(parent)
{
	addItem("string");
	addItem("number_float");
	addItem("number_double");
	addItem("number_int8");
	addItem("number_uint8");
	addItem("number_int16");
	addItem("number_uint16");
	addItem("number_int32");
	addItem("number_uint32");
	addItem("number_int64");
	addItem("number_uint64");
	addItem("time");
	addItem("duration");
	addItem("ipv4");
	addItem("ipv6");
	addItem("mac_address");
}
