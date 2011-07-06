/*
 * $Id: pvargument.h 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#ifndef PVCORE_PVARGUMENT_H
#define PVCORE_PVARGUMENT_H

#include <QString>
#include <QList>

#include <pvcore/general.h>

namespace PVCore {

  class LibExport PVArgument {
  public:

    enum pvargument_type {
      NONE,

      AXES,
      AXIS,
      CALENDAR,
      COLOR_PICKER,
      DOUBLE,
      DUALSLIDER,
      FLOAT,
      INTEGER,
      LAYER,
      SINGLESLIDER,
      STRING,
      TRIPLESLIDER,
    };



    PVArgument();
    ~PVArgument();

    QString get_name() const;
    pvargument_type get_type() const;

    void set_name(QString name);
    void set_type(pvargument_type type);

  protected:
    QString name;
    pvargument_type type;
  };


  class PVStringArgument : public PVArgument {
  private:
    QString value;
  public:
    PVStringArgument();
    ~PVStringArgument();

    QString get_value() const;

    void set_value(QString val);
  };

  class PVDualSliderArgument : public PVArgument {
  private:
    float a;
    float b;
    float min;
    float max;
  public:
    PVDualSliderArgument(float minimum, float maximum);
    ~PVDualSliderArgument();

    QList<float> get_value() const;
    
    void set_value(QList<float> val);
  };





}

#endif	/* PVCORE_PVARGUMENT_H */
