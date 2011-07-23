#ifndef PVFIELDSPLITTERCSVPARAMWIDGET_H
#define PVFIELDSPLITTERCSVPARAMWIDGET_H

#include <pvcore/general.h>
#include <pvfilter/PVFieldsFilterParamWidget.h>
#include <boost/shared_ptr.hpp>

#include <QBoxLayout>
#include <QLabel>
#include <QObject>
#include <QAction>
#include <QLineEdit>

namespace PVFilter {

class PVFieldSplitterRegexpParamWidget: public QObject, public PVFieldsSplitterParamWidget {
    Q_OBJECT;
public:
    PVFieldSplitterRegexpParamWidget();

    CLASS_REGISTRABLE_NOCOPY(PVFieldSplitterRegexpParamWidget)
};

}

#endif
