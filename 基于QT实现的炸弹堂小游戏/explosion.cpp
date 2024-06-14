#include "explosion.h"
#include <QPixmap>
#include <QGraphicsScene>

Explosion::Explosion(QGraphicsScene* scene, QGraphicsItem* parent)
    : QObject(), QGraphicsPixmapItem(parent)
{
    QPixmap explosionImage(":/img/explosion.png");
    setPixmap(explosionImage.scaled(50, 50));

    timer = new QTimer(this);
    connect(timer, &QTimer::timeout, this, &Explosion::handleExpire);
    timer->start(1000);
}

void Explosion::handleExpire()
{
    scene()->removeItem(this);
    emit expired();
    deleteLater();
    delete this;
}

