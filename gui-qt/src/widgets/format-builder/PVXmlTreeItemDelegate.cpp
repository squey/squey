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

#include <PVXmlTreeItemDelegate.h>

/******************************************************************************
 *
 * PVInspector::PVXmlTreeItemDelegate::PVXmlTreeItemDelegate
 *
 *****************************************************************************/
PVInspector::PVXmlTreeItemDelegate::PVXmlTreeItemDelegate() : QAbstractItemDelegate()
{
	setObjectName("PVXmlTreeItemDelegate");
}

/******************************************************************************
 *
 * PVInspector::PVXmlTreeItemDelegate::~PVXmlTreeItemDelegate
 *
 *****************************************************************************/
PVInspector::PVXmlTreeItemDelegate::~PVXmlTreeItemDelegate() = default;

// void MyItemDelegate::paint(QPainter *painter, const QStyleOptionViewItem &option, const
// QModelIndex &index) const {
//
//    if(true){
//
//    }
//    painter->drawRect(option.rect.topLeft().x(), option.rect.topLeft().y(), 60, 25);
//    //painter->drawRect(option.rect);
//    //painter->drawRect(5, 5, 60, 250);
//
//    QString text = ((NodeDom*)(index.internalPointer()))->getName();
//    painter->drawText(option.rect.topLeft().x()+3, option.rect.topLeft().y()+17, text);
//}

/******************************************************************************
 *
 *  PVInspector::PVXmlTreeItemDelegate::sizeHint
 *
 *****************************************************************************/
QSize PVInspector::PVXmlTreeItemDelegate::sizeHint(const QStyleOptionViewItem& /*option*/,
                                                   const QModelIndex& /*index*/) const
{
	return QSize(300, 30);
}
