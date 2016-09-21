/**
 * @file
 *
 * @copyright (C) Picviz Labs 2012-March 2015
 * @copyright (C) ESI Group INENDI April 2015-2015
 */

#ifndef MODEL_H
#define MODEL_H

#include <QString>
#include <QVector>

#include <math.h>

#include "pvguiqt/PVPoint3D.h"

namespace PVGuiQt
{

class PVLogoModel
{
  public:
	PVLogoModel() {}
	explicit PVLogoModel(const QString& filePath);

	void render(bool wireframe = false, bool normals = false) const;

	QString fileName() const { return m_fileName; }
	int faces() const { return m_pointIndices.size() / 3; }
	int edges() const { return m_edgeIndices.size() / 2; }
	int points() const { return m_points.size(); }

  private:
	QString m_fileName;
	QVector<PVGuiQt::PVPoint3D> m_points;
	QVector<PVGuiQt::PVPoint3D> m_normals;
	QVector<int> m_edgeIndices;
	QVector<int> m_pointIndices;
};
}

#endif
