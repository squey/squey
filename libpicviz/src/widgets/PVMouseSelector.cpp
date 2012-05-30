/**
 *
 * This file is part of Tulip (www.tulip-software.org)
 *
 * Authors: David Auber and the Tulip development Team
 * from LaBRI, University of Bordeaux 1 and Inria Bordeaux - Sud Ouest
 *
 * Tulip is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation, either version 3
 * of the License, or (at your option) any later version.
 *
 * Tulip is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 */

#ifdef  _WIN32
// compilation pb workaround
#include <windows.h>
#endif

#include <QEvent>
#include <QMouseEvent>

#include <tulip/Graph.h>
#include <tulip/BooleanProperty.h>
#include <tulip/LayoutProperty.h>
#include <tulip/GlMainWidget.h>
#include <tulip/GlTools.h>
#include <picviz/widgets/PVMouseSelector.h>

using namespace std;
using namespace tlp;

PVMouseSelector::PVMouseSelector(Qt::MouseButton button, Qt::KeyboardModifier modifier):
  mButton(button), kModifier(modifier), x(0),y(0),w(0),h(0),
  started(false),graph(0), _mode(EdgesAndNodes) {
}
//==================================================================
PVMouseSelector::PVMouseSelector(Qt::MouseButton button,
                             Qt::KeyboardModifier modifier, SelectionMode mode):
  mButton(button), kModifier(modifier), x(0),y(0),w(0),h(0),
  started(false),graph(0), _mode(mode) {
}
//==================================================================
bool PVMouseSelector::eventFilter(QObject *widget, QEvent *e) {
  GlMainWidget *glMainWidget = static_cast<GlMainWidget *>(widget);
  QMouseEvent * qMouseEv = static_cast<QMouseEvent *>(e);

  if (e->type() == QEvent::MouseButtonPress) {

	  BooleanProperty* selection = glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelected();
	  selection->setAllNodeValue(false);
	  selection->setAllEdgeValue(false);

	  int clampedX=qMouseEv->x();
	  int clampedY=qMouseEv->y();

	  if(clampedX<0)
		clampedX=0;

	  if(clampedY<0)
		clampedY=0;

	  if(clampedX>glMainWidget->width())
		clampedX=glMainWidget->width();

	  if(clampedY>glMainWidget->height())
		clampedY=glMainWidget->height();

	  int x = qMouseEv->x();
	  int y = qMouseEv->y();

	  int w = clampedX - x;
	  int h = clampedY - y;

      node tmpNode;
      edge tmpEdge;
      ElementType type;
	  bool hoveringOverNode = glMainWidget->doSelect(x, y, type, tmpNode, tmpEdge) && type == tlp::NODE;
	  bool hoveringOverEdge = glMainWidget->doSelect(x, y, type, tmpNode, tmpEdge) && type == tlp::EDGE;

	  if (hoveringOverNode){
		  selection->setNodeValue(tmpNode, true);
	  }
	  if (hoveringOverEdge){
		  selection->setEdgeValue(tmpEdge, true);
	  }

	  return true;
  }

  return false;
}
//==================================================================
bool PVMouseSelector::draw(GlMainWidget *glMainWidget) {
  if (!started) return false;

  if (glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph()!=graph) {
    graph = 0;
    started = false;
  }

  float yy = glMainWidget->height() - y;
  glPushAttrib(GL_ALL_ATTRIB_BITS);
  glMatrixMode (GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity ();
  gluOrtho2D (0.0, (GLdouble) glMainWidget->width(), 0.0, (GLdouble) glMainWidget->height());
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  glDisable(GL_LIGHTING);
  glDisable(GL_CULL_FACE);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA,GL_SRC_COLOR);
  float col[4]= {0,0,0,0.2f};

  if (mousePressModifier ==
#if defined(__APPLE__)
      Qt::AltModifier
#else
      Qt::ControlModifier
#endif
     ) {
    col[0]=1.f;
    col[1]=0.8f;
    col[2]=1.f;
  }
  else if(mousePressModifier == Qt::ShiftModifier) {
    col[0]=1.f;
    col[1]=.7f;
    col[2]=.7f;
  }
  else {
    col[0]=0.8f;
    col[1]=0.8f;
    col[2]=0.7f;
  }

  setColor(col);
  glBegin(GL_QUADS);
  glVertex2f(x, yy);
  glVertex2f(x+w, yy);
  glVertex2f(x+w, yy-h);
  glVertex2f(x, yy-h);
  glEnd();
  glDisable(GL_BLEND);
  glLineWidth(2);
  glLineStipple(2, 0xAAAA);
  glEnable(GL_LINE_STIPPLE);
  glBegin(GL_LINE_LOOP);
  glVertex2f(x, yy);
  glVertex2f(x+w, yy);
  glVertex2f(x+w, yy-h);
  glVertex2f(x, yy-h);
  glEnd();
  glLineWidth(1);
  glPopMatrix();
  glMatrixMode (GL_PROJECTION);
  glPopMatrix();
  glMatrixMode (GL_MODELVIEW);
  glPopAttrib();
  return true;
}
