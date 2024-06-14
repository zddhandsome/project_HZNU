#include "destructiblewall.h"

DestructibleWall::DestructibleWall(QGraphicsItem *parent)
    : QGraphicsPixmapItem(parent)
{
    QPixmap wallImage(":/img/brick.png");
    setPixmap(wallImage.scaled(50, 50));
}
