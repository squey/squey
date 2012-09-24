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
    PVLogoModel(const QString &filePath);

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
