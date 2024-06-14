#ifndef PLAYER_H
#define PLAYER_H

#include <QObject>
#include <QGraphicsPixmapItem>
#include <QKeyEvent>
#include "indestructiblewall.h"
#include "destructiblewall.h"
#include "bomb.h"
#include "explosion.h"

// Forward declaration of QGraphicsScene
class QGraphicsScene;

class Player : public QObject, public QGraphicsPixmapItem
{
    Q_OBJECT
public:
    explicit Player(int playerNumber, int monsterNumber, QGraphicsScene* scene, QGraphicsItem *parent = nullptr);
    void keyPressEvent(QKeyEvent *event) override;
    void placeBomb();
    QString getPlayerName();

signals:
    void playerMoved(int newX, int newY, int playerNumber);

    void bombPlaced(Bomb* bomb);

private:
    void movePlayer(int dx, int dy);
    int playerNumber; // Player number
};

#endif // PLAYER_H
