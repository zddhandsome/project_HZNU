#include "indestructiblewall.h"

IndestructibleWall::IndestructibleWall(int row, int column, QGraphicsItem* parent)
    : QGraphicsPixmapItem(parent)
{
    QPixmap wallImage(":/img/wall.png");
    setPixmap(wallImage.scaled(50, 50));

    int size = 50;
    int x = column * size;
    int y = row * size;
    setPos(x, y);
}
