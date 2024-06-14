#ifndef INDESTRUCTIBLEWALL_H
#define INDESTRUCTIBLEWALL_H

#include <QGraphicsPixmapItem>

class IndestructibleWall : public QGraphicsPixmapItem
{
public:
    IndestructibleWall(int row, int column, QGraphicsItem* parent = nullptr);
};

#endif // INDESTRUCTIBLEWALL_H

