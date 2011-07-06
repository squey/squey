/*
 * $Id: pvargument.cpp 3090 2011-06-09 04:59:46Z stricaud $
 * Copyright (C) Sebastien Tricaud 2010-2011
 * Copyright (C) Philippe Saade 2010-2011
 * 
 */

#include <pvcore/pvargument.h>

PVCore::PVArgument::PVArgument() {
  name = QString("");
  type = PVCore::PVArgument::NONE;
}

PVCore::PVArgument::~PVArgument() {

}

QString PVCore::PVArgument::get_name() const 
{
  return name;
}

PVCore::PVArgument::pvargument_type PVCore::PVArgument::get_type() const 
{
  return type;
}

void PVCore::PVArgument::set_name(QString n)
{
  name = n;
}

void PVCore::PVArgument::set_type(PVCore::PVArgument::pvargument_type t)
{
  type = t;
}

PVCore::PVStringArgument::PVStringArgument()
{
  set_type(PVCore::PVArgument::STRING);
}

PVCore::PVStringArgument::~PVStringArgument()
{
}

QString PVCore::PVStringArgument::get_value() const
{
  return value;
}

void PVCore::PVStringArgument::set_value(QString val)
{
  value = val;
}


PVCore::PVDualSliderArgument::PVDualSliderArgument(float minimum = 0.0, float maximum = 1.0)
{
  set_type(PVCore::PVArgument::DUALSLIDER);

  min = minimum;
  max = maximum;
}

PVCore::PVDualSliderArgument::~PVDualSliderArgument()
{

}

QList<float> PVCore::PVDualSliderArgument::get_value() const
{
  QList<float> values;

  values[0] = a;
  values[1] = b;

  return values;
}

void PVCore::PVDualSliderArgument::set_value(QList<float> val)
{
  a = val[0];
  b = val[1];
}
