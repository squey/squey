/*
* Copyright (C) 2009 Matteo Bertozzi.
*
* This file is part of THLibrary.
*
* THLibrary is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* THLibrary is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with THLibrary. If not, see <http://www.gnu.org/licenses/>.
*/

#include <QWebFrame>
#include <QTimer>
#include <QFile>

#include <QDebug>

#include <geo/GKMapView_p.h>
#include <geo/GKMapView.h>

/* ===========================================================================
 *  PRIVATE Class
 */
GKMapViewJsObject::GKMapViewJsObject (QObject *parent)
    : QObject(parent)
{
    m_isLoaded = false;

    QWebView *webView = static_cast<QWebView *>(parent);
    connect(webView, SIGNAL(loadFinished(bool)), this, SLOT(loadFinished(bool)));

    QTimer::singleShot(0, this, SLOT(loadMapsPage()));
}

GKMapViewJsObject::~GKMapViewJsObject() {
}

void GKMapViewJsObject::addJsRequest (const QString& scriptSource) {
    if (m_isLoaded) 
        evaluateJavaScript(scriptSource);
    else
        m_jsRequests.enqueue(scriptSource);
}

QVariant GKMapViewJsObject::evaluateJavaScript (const QString& scriptSource) {
    return(webFrame()->evaluateJavaScript(scriptSource));
}

void GKMapViewJsObject::addToJavaScriptWindowObject (const QString& name, QObject *object) 
{
    webFrame()->addToJavaScriptWindowObject(name, object);
}

QWebView *GKMapViewJsObject::webView (void) const {
    return(static_cast<QWebView *>(parent()));
}

QWebPage *GKMapViewJsObject::webPage (void) const {
    return(static_cast<QWebView *>(parent())->page());
}

QWebFrame *GKMapViewJsObject::webFrame (void) const {
    return(static_cast<QWebView *>(parent())->page()->currentFrame());
}

GKMapView *GKMapViewJsObject::mapView (void) const {
    return(static_cast<GKMapView *>(parent()));
}

void GKMapViewJsObject::jsLocationClicked (qreal latitude, qreal longitude) {
    qDebug() << "JS Location Clicked" << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->locationClicked(coordinate);
}

void GKMapViewJsObject::jsLocationRightClicked (qreal latitude, qreal longitude) {
    qDebug() << "JS Location Right Clicked" << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->locationRightClick(coordinate);
}

void GKMapViewJsObject::jsCenterLocationChanged (qreal latitude, qreal longitude) {
    qDebug() << "JS Center Location Changed" << latitude << longitude;

    centerCoordinates.latitude = latitude;
    centerCoordinates.latitude = longitude;
    emit mapView()->centerLocationChanged(centerCoordinates);
}

void GKMapViewJsObject::jsMarkerClicked (const QString& title, qreal latitude, qreal longitude) 
{
    qDebug() << "JS Marker Clicked" << title << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->markerClicked(title, coordinate);
}

void GKMapViewJsObject::jsAddressLocation (const QString& address, 
                                           qreal latitude, qreal longitude)
{
    qDebug() << "Address Location" << address << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->addressFound(address, coordinate);
}

void GKMapViewJsObject::jsAddressNotFound (const QString& address) {
    qDebug() << "JS Address Not Found" << address;
    emit mapView()->addressNotFound(address);
}

void GKMapViewJsObject::jsLocationAddress (const QString& address, 
                                           qreal latitude, qreal longitude)
{
    qDebug() << "JS Location Address" << address << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->locationFound(address, coordinate);
}

void GKMapViewJsObject::jsLocationNotFound (qreal latitude, qreal longitude) {
    qDebug() << "JS Location Not Found" << latitude << longitude;

    GKLocationCoordinate2D coordinate;
    coordinate.latitude = latitude;
    coordinate.longitude = longitude;
    emit mapView()->locationNotFound(coordinate);
}

void GKMapViewJsObject::loadMapsPage (void) {
    QFile file(":/gmaps.html");
    if (file.open(QIODevice::ReadOnly)) {
        QByteArray mapHtml = file.readAll();
        file.close();

        webView()->setHtml(mapHtml);
    }
}

void GKMapViewJsObject::loadFinished (bool ok) {
    if (ok) {
        m_isLoaded = true;

        addToJavaScriptWindowObject("gkMapView", this);

        QWebFrame *frame = webFrame();
        frame->evaluateJavaScript("initialize()");
        while (!m_jsRequests.isEmpty())
            frame->evaluateJavaScript(m_jsRequests.dequeue());
    }
}

/* ===========================================================================
 *  PUBLIC GKMapView Class
 */
GKMapView::GKMapView (QWidget *parent)
    : QWebView(parent), d(new GKMapViewJsObject(this))
{
}

GKMapView::~GKMapView() {
    delete d;
}

/* Returns the Map Type */
GKMapType GKMapView::mapType (void) const {
    return(d->mapType);
}

/**
 * Set Map Type: Satellite, Hybrid, Terrain, Roadmap.
 */
void GKMapView::setMapType (GKMapType mapType) {
    switch ((d->mapType = mapType)) {
        case GKMapTypeSatellite:
            d->addJsRequest("switchToSatellite()");
            break;
        case GKMapTypeTerrain:
            d->addJsRequest("switchToTerrain()");
            break;
        case GKMapTypeHybrid:
            d->addJsRequest("switchToHybrid()");
            break;
        case GKMapRoadmap:
            d->addJsRequest("switchToRoadmap()");
            break;
    }
}

/**
 * Returns the center Coordinate of the map
 */
GKLocationCoordinate2D GKMapView::centerCoordinate (void) const {
    return(d->centerCoordinates);
}

/**
 * Set Central location of the Map, using the specified address.
 */
void GKMapView::setLocation (const QString& address) {
    d->addJsRequest(QString("setLocationFromAddress('%1')").arg(address));
}

/**
 * Set Central location of the Map.
 */
void GKMapView::setLocation (GKLocationDegrees latitude,
                             GKLocationDegrees longitude)
{
    d->addJsRequest(QString("setLocation(%1, %2)").arg(latitude).arg(longitude));
}

/**
 * Set Central location of the Map.
 */
void GKMapView::setLocation (const GKLocationCoordinate2D& coordinate) {
    setLocation(coordinate.latitude, coordinate.longitude);
}

/**
 * Add Marker at specified location, 
 * you can get the markerClicked() signal to interact with it.
 */
void GKMapView::addMarker (const QString& title,
                           GKLocationDegrees latitude,
                           GKLocationDegrees longitude)
{
    d->addJsRequest(QString("addMarker('%1', %2, %3)").arg(title).arg(latitude).arg(longitude));
}

/**
 * Add Marker at specified location, 
 * you can get the markerClicked() signal to interact with it.
 */
void GKMapView::addMarker (const QString& title,
                           const GKLocationCoordinate2D& coordinate)
{
    addMarker(title, coordinate.latitude, coordinate.longitude);
}

/**
 * Add Marker at specified Address, 
 * you can get the markerClicked() signal to interact with it.
 */
void GKMapView::addMarker (const QString& title, const QString& address) {
    d->addJsRequest(QString("addMarkerAtAddress('%1', '%2')").arg(title).arg(address));
}

/**
 * Map Zoom In
 */
void GKMapView::mapZoomIn (void) {
    d->addJsRequest("zoomIn()");
}

/**
 * Map Zoom Out
 */
void GKMapView::mapZoomOut (void) {
    d->addJsRequest("zoomOut()");
}

/**
 * Set Map Zoom Factor
 */
void GKMapView::setMapZoom (qreal zoom) {
    d->addJsRequest(QString("zoom(%1)").arg(zoom));
}

/**
 * Add marker and Info window, with your 'contentString', at specified address.
 */
void GKMapView::addMarkerWithWindow (const QString& title,
                                     const QString& contentString,
                                     const QString& address)
{
    d->addJsRequest(QString("addMarkerAtAddressWithWindow('%1', '%2', '%3')").arg(title).arg(contentString).arg(address));
}

/**
 * Add marker and Info window, with your 'contentString', at specified coordinates.
 */
void GKMapView::addMarkerWithWindow (const QString& title,
                                     const QString& contentString,
                                     GKLocationDegrees latitude,
                                     GKLocationDegrees longitude)
{
    d->addJsRequest(QString("addMarkerWindow('%1', '%2', %3, %4)").arg(title).arg(contentString).arg(latitude).arg(longitude));
}

/**
 * Add marker and Info window, with your 'contentString', at specified coordinates.
 */
void GKMapView::addMarkerWithWindow (const QString& title,
                                    const QString& contentString,
                                    const GKLocationCoordinate2D& coordinate)
{
    addMarkerWithWindow(title, contentString, coordinate.latitude, coordinate.longitude);
}

/**
 * Remove marker at specified location address.
 */
void GKMapView::removeMarker (const QString& address) {
    d->addJsRequest(QString("removeMarkerFromAddress('%1')").arg(address));
}

/**
 * Remove marker at specified location coordinates.
 */
void GKMapView::removeMarker (const GKLocationCoordinate2D& coordinate) {
    removeMarker(coordinate.latitude, coordinate.longitude);
}

/**
 * Remove marker at specified location coordinates.
 */
void GKMapView::removeMarker (GKLocationDegrees latitude,
                              GKLocationDegrees longitude)
{
    d->addJsRequest(QString("removeMarker(%1, %2)").arg(latitude).arg(longitude));
}

/**
 * Add Info window, with your 'contentString', at specified address.
 */
void GKMapView::addInfoWindow (const QString& contentString,
                               const QString& address)
{
    d->addJsRequest(QString("addInfoWindowAtAddress('%1', '%2')").arg(contentString).arg(address));
}

/**
 * Add Info window, with your 'contentString', at specified coordinates.
 */
void GKMapView::addInfoWindow (const QString& contentString,
                               GKLocationDegrees latitude,
                               GKLocationDegrees longitude)
{
    d->addJsRequest(QString("addInfoWindow('%1', %2, %3)").arg(contentString).arg(latitude).arg(longitude));
}

/**
 * Add Info window, with your 'contentString', at specified coordinates.
 */
void GKMapView::addInfoWindow (const QString& contentString,
                               const GKLocationCoordinate2D& coordinate)
{
    addInfoWindow(contentString, coordinate.latitude, coordinate.longitude);
}

/**
 * Get the Address from specified location coordinates.
 * you need to connect to the Signals addressFound() and addressNotFound()
 * to get your results.
 */
void GKMapView::addressFromLocation (GKLocationDegrees latitude,
                                     GKLocationDegrees longitude)
{
    d->addJsRequest(QString("addressFromLocation(%1, %2)").arg(latitude).arg(longitude));
}

/**
 * Get the Address from specified location coordinates.
 * you need to connect to the Signals addressFound() and addressNotFound()
 * to get your results.
 */
void GKMapView::addressFromLocation (const GKLocationCoordinate2D& coordinate) {
    addressFromLocation(coordinate.latitude, coordinate.longitude);
}

/**
 * Get the Coordinates from specified location Address.
 * you need to connect to the Signals locationFound() and locationNotFound()
 * to get your results.
 */
void GKMapView::locationFromAddress (const QString& address) {
    d->addJsRequest(QString("addressToLocation('%1')").arg(address));
}

