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

#include <cmath>
#include <limits.h>

#include <tulip/Graph.h>
#include <tulip/LayoutProperty.h>
#include <tulip/BooleanProperty.h>
#include <tulip/DoubleProperty.h>
#include <tulip/SizeProperty.h>
#include <tulip/GlMainWidget.h>
#include <tulip/DrawingTools.h>
#include <tulip/ForEach.h>

#include <picviz/widgets/PVMouseSelectionEditor.h>

#include <QtGui/qevent.h>

#include <pvkernel/core/general.h>

#define EPSILON 1.0
#define EPSILON_SCREEN 50
#define EPSILON_STRETCH_MIN 1 - 1.0e-01
#define EPSILON_STRETCH_MAX 1 + 1.0e-01

using namespace tlp;
using namespace std;

const unsigned int arrowWithLineSize=8;
const Coord arrowWithLine[] = {Coord(0,3,0),Coord(-5,-5,0),Coord(5,-5,0),Coord(0,3,0),Coord(5,3,0),Coord(5,5,0),Coord(-5,5,0),Coord(-5,3,0)};
const unsigned int twoArrowWithLineSize=10;
const Coord twoArrowWithLine[] = {Coord(0,0,0),Coord(5,-5,0),Coord(-5,-5,0),Coord(0,0,0),Coord(-5,0,0),Coord(5,0,0),Coord(0,0,0),Coord(5,5,0),Coord(-5,5,0),Coord(0,0,0)};

//========================================================================================
PVMouseSelectionEditor::PVMouseSelectionEditor():glMainWidget(NULL),layer(NULL),composite(NULL),_started(true) {
  operation = NONE;

  centerRect.setStencil(0);
  advRect.setStencil(0);

  Color hudColor(128,128,128,128);
  centerRect.setFillMode(true);
  centerRect.setOutlineMode(true);
  centerRect.setFillColor(hudColor);
  hudColor=Color(128,128,128,64);
  advRect.setFillMode(true);
  advRect.setOutlineMode(false);
  advRect.setFillColor(hudColor);
}
//========================================================================================
PVMouseSelectionEditor::~PVMouseSelectionEditor() {
  if(layer) {
    glMainWidget->getScene()->removeLayer(layer,true);
    layer=NULL;
  }
}
//========================================================================================
void PVMouseSelectionEditor::getOperation(GlEntity *select) {
  if( select == &_advControls[0]) {
    operation = ALIGN_TOP;
    return;
  }

  if( select == &_advControls[1]) {
    operation = ALIGN_BOTTOM;
    return;
  }

  if( select == &_advControls[2]) {
    operation = ALIGN_LEFT;
    return;
  }

  if( select == &_advControls[3]) {
    operation = ALIGN_RIGHT;
    return;
  }

  if( select == &_advControls[4]) {
    operation = ALIGN_HORIZONTALLY;
    return;
  }

  if( select == &_advControls[5]) {
    operation = ALIGN_VERTICALLY;
    return;
  }
}
//========================================================================================
bool PVMouseSelectionEditor::eventFilter(QObject *widget, QEvent *e) {

	  QMouseEvent * qMouseEv = (QMouseEvent *) e;
	  GlMainWidget *glMainWidget = (GlMainWidget *) widget;


  if (e->type() == QEvent::MouseMove) {

	  if(_started) {
			 int x = qMouseEv->x();
		int y = qMouseEv->y();


		  mMouseTranslate(x, y, glMainWidget);

		  glMainWidget->draw();
		  return true;
	  }
	  else {
		  //_started = true;
	  }


  }
  else if (e->type() == QEvent::MouseButtonRelease)
  {
	  PVLOG_INFO("PVMouseSelectionEditor::eventFilter : QEvent::MouseButtonRelease\n");
	  BooleanProperty* selection = glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getElementSelected();
	  	  selection->setAllNodeValue(false);
	  	  selection->setAllEdgeValue(false);
		GlMainWidget *glMainWidget = (GlMainWidget *) widget;
		stopEdition();
		glMainWidget->draw();
		_started = false;
		return true;

  }

  return true;
}
//========================================================================================
bool PVMouseSelectionEditor::compute(GlMainWidget *glMainWidget) {
  if (computeFFD(glMainWidget)) {
    if(!layer) {
      layer=new GlLayer("selectionEditorLayer",true);
      layer->setCamera(Camera(glMainWidget->getScene(),false));
      glMainWidget->getScene()->insertLayerAfter(layer,"Main");
      composite = new GlComposite(false);
      layer->addGlEntity(composite,"selectionComposite");
    }

    composite->addGlEntity(&centerRect, "CenterRectangle");

    Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
    int moreThanOneNode=0;

    while(itN->hasNext()) {
      if(moreThanOneNode>=2)
        break;

      moreThanOneNode++;
      itN->next();
    }

    delete itN;

    if(moreThanOneNode>=2) {
      composite->addGlEntity(&advRect, "AdvRectangle");

      composite->addGlEntity(&_advControls[0], "center-top");
      composite->addGlEntity(&_advControls[1], "center-bottom");
      composite->addGlEntity(&_advControls[2], "center-right");
      composite->addGlEntity(&_advControls[3], "center-left");
      composite->addGlEntity(&_advControls[4], "center-horizontally");
      composite->addGlEntity(&_advControls[5], "center-vertically");
    }
    else {
      composite->deleteGlEntity("AdvRectangle");

      composite->deleteGlEntity("center-top");
      composite->deleteGlEntity("center-bottom");
      composite->deleteGlEntity("center-right");
      composite->deleteGlEntity("center-left");
      composite->deleteGlEntity("center-horizontally");
      composite->deleteGlEntity("center-vertically");
    }

    this->glMainWidget=glMainWidget;
    return true;
  }
  else {
    if(layer) {
      glMainWidget->getScene()->removeLayer(layer,true);
      layer=NULL;
    }

    return false;
  }
}
//========================================================================================
bool PVMouseSelectionEditor::draw(GlMainWidget *) {
  //return compute(glMainWidget);
  return true;
}
//========================================================================================
void PVMouseSelectionEditor::initEdition() {
  _graph->push();
}
//========================================================================================
void PVMouseSelectionEditor::undoEdition() {
  if (operation == NONE) return;

  _graph->pop();
  operation = NONE;
}
//========================================================================================
void PVMouseSelectionEditor::stopEdition() {
  //cerr << __PRETTY_FUNCTION__ << endl;
  if(layer) {
    glMainWidget->getScene()->removeLayer(layer,true);
    layer=NULL;
  }

  operation = NONE;
}
//========================================================================================
void PVMouseSelectionEditor::initProxies(GlMainWidget *glMainWidget) {
  GlGraphInputData *inputData=glMainWidget->getScene()->getGlGraphComposite()->getInputData();
  _graph     = inputData->getGraph();
  _layout    = inputData->getLayoutProperty();
  _selection = _graph->getProperty<BooleanProperty>(inputData->getElementSelectedPropName());
  _rotation  = _graph->getProperty<DoubleProperty>(inputData->getElementRotationPropName());
  _sizes     = _graph->getProperty<SizeProperty>(inputData->getElementSizePropName());
}
//========================================================================================
void PVMouseSelectionEditor::mMouseTranslate(double newX, double newY, GlMainWidget *glMainWidget) {
  //  cerr << __PRETTY_FUNCTION__ << endl;
  Observable::holdObservers();
  initProxies(glMainWidget);
  Coord v0(0,0,0);
  Coord v1((double)(editPosition[0] - newX), -(double)(editPosition[1] - newY),0);
  v0 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v0);
  v1 = glMainWidget->getScene()->getCamera().screenTo3DWorld(v1);
  v1 -= v0;
  Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
  Iterator<edge> *itE = _selection->getEdgesEqualTo(true, _graph);
  _layout->translate(v1, itN, itE);
  delete itN;
  delete itE;
  editPosition[0]  = newX;
  editPosition[1]  = newY;
  Observable::unholdObservers();
}
//========================================================================================
void PVMouseSelectionEditor::mMouseStretchAxis(double newX, double newY, GlMainWidget* ) {
  //  cerr << __PRETTY_FUNCTION__ << "/op=" << operation << ", mod:" << mode << endl;
  Coord curPos(newX, newY, 0);
  Coord stretch(1,1,1);

  //  cerr << "cur : << " << curPos << " center : " << editCenter << endl;
  if (operation == STRETCH_X || operation == STRETCH_XY) {
    stretch[0] =  (curPos[0] - editCenter[0]) / (editPosition[0] - editCenter[0]);
  }

  if (operation == STRETCH_Y || operation == STRETCH_XY) {
    stretch[1] = (curPos[1] - editCenter[1]) / (editPosition[1] - editCenter[1]);
  }

  //  cerr << "stretch : << "<< stretch << endl;

  Observable::holdObservers();
  _graph->pop();
  _graph->push();

  //stretch layout
  if (mode == COORD_AND_SIZE || mode == COORD) {
    Coord center(editLayoutCenter);
    center *= -1.;
    //move the center to the origin in order to be able to scale
    Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
    Iterator<edge> *itE = _selection->getEdgesEqualTo(true, _graph);
    _layout->translate(center, itN, itE);
    delete itN;
    delete itE;
    //scale the drawing
    itN = _selection->getNodesEqualTo(true, _graph);
    itE = _selection->getEdgesEqualTo(true, _graph);
    _layout->scale(stretch, itN, itE);
    delete itN;
    delete itE;
    //replace the center of the graph at its originale position
    center *= -1.;
    itN = _selection->getNodesEqualTo(true, _graph);
    itE = _selection->getEdgesEqualTo(true, _graph);
    _layout->translate(center, itN, itE);
    delete itN;
    delete itE;
  }

  //stretch size
  if (mode == COORD_AND_SIZE || mode == SIZE) {
    Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
    Iterator<edge> *itE = _selection->getEdgesEqualTo(true, _graph);
    _sizes->scale(stretch, itN, itE);
    delete itN;
    delete itE;
  }

  Observable::unholdObservers();
}
//========================================================================================
void PVMouseSelectionEditor::mMouseRotate(double newX, double newY, GlMainWidget *glMainWidget) {
  //  cerr << __PRETTY_FUNCTION__ << endl;
  if (operation == ROTATE_Z) {
    Coord curPos(newX, newY, 0);

    Coord vCS = editPosition - editCenter;
    vCS /= vCS.norm();
    Coord vCP =  curPos - editCenter;
    vCP /= vCP.norm();

    float sign = (vCS ^ vCP)[2];
    sign /= fabs(sign);
    double cosalpha = vCS.dotProduct(vCP);
    double deltaAngle = sign * acos(cosalpha);

    Observable::holdObservers();

    initProxies(glMainWidget);
    _graph->pop();
    _graph->push();

    double degAngle = (deltaAngle * 180.0 / M_PI);

    //rotate layout
    if (mode == COORD_AND_SIZE || mode == COORD) {
      Coord center(editLayoutCenter);
      center *= -1.;
      Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
      Iterator<edge> *itE = _selection->getEdgesEqualTo(true, _graph);
      _layout->translate(center, itN, itE);
      delete itN;
      delete itE;
      itN = _selection->getNodesEqualTo(true, _graph);
      itE = _selection->getEdgesEqualTo(true, _graph);
      _layout->rotateZ(-degAngle, itN, itE);
      delete itN;
      delete itE;
      center *= -1.;
      itN = _selection->getNodesEqualTo(true, _graph);
      itE = _selection->getEdgesEqualTo(true, _graph);
      _layout->translate(center, itN, itE);
      delete itN;
      delete itE;
    }

    if (mode == COORD_AND_SIZE || mode == SIZE) {
      Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);

      while(itN->hasNext()) {
        node n = itN->next();
        double rotation = _rotation->getNodeValue(n);
        _rotation->setNodeValue(n, rotation - degAngle);
      }

      delete itN;
    }

    Observable::unholdObservers();
  }
  else {
    double initDelta, delta, cosa;
    double xAngle = 0, yAngle = 0;
    double nbPI = 0;

    delta = abs(newX - editPosition[0]);

    if (delta > abs(newY - editPosition[1])) {
      initDelta = abs(editCenter[0] - editPosition[0]);
      nbPI = floor(delta / (2. * initDelta));
      delta -= nbPI * 2. * initDelta;
      cosa = (initDelta - delta)/initDelta;

      yAngle = (acos(cosa) + (nbPI * M_PI)) * 180.0 / M_PI;
    }
    else {
      delta = abs(newY - editPosition[1]);
      initDelta = abs(editCenter[1] - editPosition[1]);
      nbPI = floor(delta / (2. * initDelta));
      delta -= nbPI * 2. * initDelta;
      cosa = (initDelta - delta)/initDelta;

      xAngle = (acos(cosa) + (nbPI * M_PI)) * 180.0 / M_PI;
    }

    Observable::holdObservers();

    initProxies(glMainWidget);
    _graph->pop();
    _graph->push();

    Coord center(editLayoutCenter);
    center *= -1.;
    Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
    Iterator<edge> *itE = _selection->getEdgesEqualTo(true, _graph);
    _layout->translate(center, itN, itE);
    delete itN;
    delete itE;
    itN = _selection->getNodesEqualTo(true, _graph);
    itE = _selection->getEdgesEqualTo(true, _graph);

    if (yAngle > xAngle)
      _layout->rotateY(yAngle, itN, itE);
    else
      _layout->rotateX(xAngle, itN, itE);

    delete itN;
    delete itE;
    center *= -1.;
    itN = _selection->getNodesEqualTo(true, _graph);
    itE = _selection->getEdgesEqualTo(true, _graph);
    _layout->translate(center, itN, itE);
    delete itN;
    delete itE;

    Observable::unholdObservers();
  }
}
//========================================================================================
void PVMouseSelectionEditor::mAlign(EditOperation operation,GlMainWidget*) {
  Observable::holdObservers();
  _graph->push();

  Iterator<node> *itN = _selection->getNodesEqualTo(true, _graph);
  bool init=false;
  float min = -FLT_MAX, max = FLT_MAX;

  while (itN->hasNext()) {
    node n=itN->next();

    float valueMin = -FLT_MAX, valueMax = FLT_MAX;

    switch(operation) {
    case ALIGN_TOP:
      valueMax=_layout->getNodeValue(n)[1]+_sizes->getNodeValue(n)[1]/2.;
      break;

    case ALIGN_BOTTOM:
      valueMin=_layout->getNodeValue(n)[1]-_sizes->getNodeValue(n)[1]/2.;
      break;

    case ALIGN_HORIZONTALLY:
      valueMax=_layout->getNodeValue(n)[1]+_sizes->getNodeValue(n)[1]/2.;
      valueMin=_layout->getNodeValue(n)[1]-_sizes->getNodeValue(n)[1]/2.;
      break;

    case ALIGN_LEFT:
      valueMin=_layout->getNodeValue(n)[0]-_sizes->getNodeValue(n)[0]/2.;
      break;

    case ALIGN_RIGHT:
      valueMax=_layout->getNodeValue(n)[0]+_sizes->getNodeValue(n)[0]/2.;
      break;

    case ALIGN_VERTICALLY:
      valueMax=_layout->getNodeValue(n)[0]-_sizes->getNodeValue(n)[0]/2.;
      valueMin=_layout->getNodeValue(n)[0]+_sizes->getNodeValue(n)[0]/2.;
      break;

    case STRETCH_X:
    case STRETCH_Y:
    case STRETCH_XY:
    case ROTATE_Z:
    case ROTATE_XY:
    case TRANSLATE:
    case NONE:
    default:
      break;
    }

    if(!init) {
      max=valueMax;
      min=valueMin;
      init=true;
    }
    else {
      switch(operation) {
      case ALIGN_TOP:
      case ALIGN_RIGHT:

        if(valueMax>max)
          max=valueMax;

        break;

      case ALIGN_BOTTOM:
      case ALIGN_LEFT:

        if(valueMin<min)
          min=valueMin;

        break;

      case ALIGN_HORIZONTALLY:
      case ALIGN_VERTICALLY:

        if(valueMax>max) max=valueMax;

        if(valueMin<min) min=valueMin;

        break;

      case STRETCH_X:
      case STRETCH_Y:
      case STRETCH_XY:
      case ROTATE_Z:
      case ROTATE_XY:
      case TRANSLATE:
      case NONE:
      default:
        break;
      }
    }
  }

  itN = _selection->getNodesEqualTo(true, _graph);

  while (itN->hasNext()) {
    node n=itN->next();
    Coord old(_layout->getNodeValue(n));

    switch(operation) {
    case ALIGN_TOP:
      old[1]=max-_sizes->getNodeValue(n)[1]/2.;
      break;

    case ALIGN_BOTTOM:
      old[1]=min+_sizes->getNodeValue(n)[1]/2.;
      break;

    case ALIGN_HORIZONTALLY:
      old[1]=(max+min)/2;
      break;

    case ALIGN_LEFT:
      old[0]=min+_sizes->getNodeValue(n)[0]/2.;
      break;

    case ALIGN_RIGHT:
      old[0]=max-_sizes->getNodeValue(n)[0]/2.;
      break;

    case ALIGN_VERTICALLY:
      old[0]=(max+min)/2;
      break;

    case STRETCH_X:
    case STRETCH_Y:
    case STRETCH_XY:
    case ROTATE_Z:
    case ROTATE_XY:
    case TRANSLATE:
    case NONE:
    default:
      break;
    }

    _layout->setNodeValue(n,old);
  }

  Observable::unholdObservers();
}
//========================================================================================
Coord minCoord(const Coord &v1, const Coord &v2) {
  Coord result;

  for (unsigned int i =0; i<3; ++i)
    result[i] = std::min(v1[i], v2[i]);

  return result;
}
Coord maxCoord(const Coord &v1, const Coord &v2) {
  Coord result;

  for (unsigned int i =0; i<3; ++i)
    result[i] = std::max(v1[i], v2[i]);

  return result;
}
//========================================================================================
//========================================================================================
bool PVMouseSelectionEditor::computeFFD(GlMainWidget *glMainWidget) {
  if (!glMainWidget->getScene()->getGlGraphComposite() || !glMainWidget->getScene()->getGlGraphComposite()->getInputData()->getGraph())
    return false;

  // We calculate the bounding box for the selection :
  initProxies(glMainWidget);
  BoundingBox boundingBox = tlp::computeBoundingBox(_graph, _layout, _sizes, _rotation, _selection);

  if (!boundingBox.isValid()) return false;

  if(operation==NONE)
    glMainWidget->setCursor(QCursor(Qt::PointingHandCursor));

  Coord min2D, max2D;
  _layoutCenter = Coord(boundingBox.center());

  //project the 8 points of the cube to obtain the bounding square on the 2D screen
  Coord bbsize(boundingBox[1] - boundingBox[0]);
  //v1
  Coord tmp(boundingBox[0]);
  tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
  min2D = tmp;
  max2D = tmp;

  //v2, v3, V4
  for (unsigned int i=0; i<3; ++i) {
    tmp = Coord(boundingBox[0]);
    tmp[i] += bbsize[i];
    tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
    min2D = minCoord(tmp, min2D);
    max2D = maxCoord(tmp, max2D);
  }

  //v4
  tmp = Coord(boundingBox[0]);
  tmp[0] += bbsize[0];
  tmp[1] += bbsize[1];
  tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
  min2D = minCoord(tmp, min2D);
  max2D = maxCoord(tmp, max2D);
  //v6
  tmp = Coord(boundingBox[0]);
  tmp[0] += bbsize[0];
  tmp[2] += bbsize[2];
  tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
  min2D = minCoord(tmp, min2D);
  max2D = maxCoord(tmp, max2D);
  //v7
  tmp = Coord(boundingBox[0]);
  tmp[1] += bbsize[1];
  tmp[2] += bbsize[2];
  tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
  min2D = minCoord(tmp, min2D);
  max2D = maxCoord(tmp, max2D);
  //v8
  tmp = Coord(boundingBox[0]);
  tmp += bbsize;
  tmp = glMainWidget->getScene()->getCamera().worldTo2DScreen(tmp);
  min2D = minCoord(tmp, min2D);
  max2D = maxCoord(tmp, max2D);

  ffdCenter = Coord(boundingBox.center());

  Coord tmpCenter = glMainWidget->getScene()->getCamera().worldTo2DScreen(ffdCenter);

  //  cerr << tmpCenter << endl;

  //tmpCenter[0] = (double)glMainWidget->width() - tmpCenter[0];
  //tmpCenter[1] = (double)glMainWidget->height() - tmpCenter[1];

  //  tmpCenter[1] = tmpCenter[1];

  int x = int(max2D[0] - min2D[0]) / 2 + 1; // (+1) because selection use glLineWidth=3 thus
  int y = int(max2D[1] - min2D[1]) / 2 + 1; //the rectangle can be too small.

  if (x < 20) x = 18;

  if (y < 20) y = 18;

  Coord positions[8];

  // we keep the z coordinate of the ffdCenter
  // to ensure a correct position of our controls (see GlHudPolygon.cpp)
  positions[0] = Coord( x,  0, ffdCenter[2]) + tmpCenter; // left
  positions[1] = Coord( x, -y, ffdCenter[2]) + tmpCenter; // Top left
  positions[2] = Coord( 0, -y, ffdCenter[2]) + tmpCenter; // Top
  positions[3] = Coord(-x, -y, ffdCenter[2]) + tmpCenter; // Top r
  positions[4] = Coord(-x,  0, ffdCenter[2]) + tmpCenter; // r
  positions[5] = Coord(-x,  y, ffdCenter[2]) + tmpCenter; // Bottom r
  positions[6] = Coord( 0,  y, ffdCenter[2]) + tmpCenter; // Bottom
  positions[7] = Coord( x,  y, ffdCenter[2]) + tmpCenter; // Bottom l

  for(int i=0; i<8; i++) {
    positions[i][2]=0;
  }

  //Parameters of the rectangle that shows the selected area.
  centerRect.setTopLeftPos(positions[1]);
  centerRect.setBottomRightPos(positions[5]);
  advRect.setTopLeftPos(positions[7]+Coord(-92,16,0));
  advRect.setBottomRightPos(positions[7]);

  vector<Coord> advControlVect;

  for(unsigned int i=0; i<arrowWithLineSize; ++i) {
    advControlVect.push_back(arrowWithLine[i]+positions[7]+Coord(-11,8,0));
  }

  _advControls[0] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<arrowWithLineSize; ++i) {
    advControlVect.push_back(Coord(arrowWithLine[i][0],-arrowWithLine[i][1],0)+positions[7]+Coord(-25,8,0));
  }

  _advControls[1] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<arrowWithLineSize; ++i) {
    advControlVect.push_back(Coord(-arrowWithLine[i][1],arrowWithLine[i][0],0)+positions[7]+Coord(-39,8,0));
  }

  _advControls[2] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<arrowWithLineSize; ++i) {
    advControlVect.push_back(Coord(arrowWithLine[i][1],arrowWithLine[i][0],0)+positions[7]+Coord(-53,8,0));
  }

  _advControls[3] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<twoArrowWithLineSize; ++i) {
    advControlVect.push_back(twoArrowWithLine[i]+positions[7]+Coord(-67,8,0));
  }

  _advControls[4] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<twoArrowWithLineSize; ++i) {
    advControlVect.push_back(Coord(twoArrowWithLine[i][1],twoArrowWithLine[i][0],0)+positions[7]+Coord(-81,8,0));
  }

  _advControls[5] = GlComplexPolygon(advControlVect,Color(255,40,40,200),Color(128,20,20,200));
  advControlVect.clear();

  for(unsigned int i=0; i<6; ++i) {
    _advControls[i].setStencil(0);
  }

  return true;
}
//========================================================================================
