#ifndef EXPLOSION_H
#define EXPLOSION_H

#include <QObject>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QTimer>

class QGraphicsScene;
class DestructibleWall;

class Explosion : public QObject, public QGraphicsPixmapItem
{
    Q_OBJECT

public:
    explicit Explosion(QGraphicsScene* scene, QGraphicsItem* parent = nullptr);

private:
    QTimer* timer;

signals:
    void expired();

public slots:
    void handleExpire();
};

#endif // EXPLOSION_H

