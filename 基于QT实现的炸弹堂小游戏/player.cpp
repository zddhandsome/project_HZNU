#include "player.h"
#include <QGraphicsScene>

Player::Player(int playerNumber, int monsterNumber, QGraphicsScene* scene, QGraphicsItem *parent)
    : QObject(), QGraphicsPixmapItem(parent), playerNumber(playerNumber)
{
    QPixmap playerImage(":/img/player" + QString::number(monsterNumber) + ".png");
    setPixmap(playerImage.scaled(50, 50));
    setFlag(QGraphicsItem::ItemIsFocusable);
    setFlag(QGraphicsItem::ItemIsMovable);
    setFocus();
    if (scene) scene->addItem(this);
}

QString Player::getPlayerName()
{
    return "Player" + QString::number(playerNumber);
}

void Player::keyPressEvent(QKeyEvent *event)
{
    int dx = 0;
    int dy = 0;

    if (playerNumber == 1) {
        switch (event->key()) {
        case Qt::Key_J:
            dx = -50;
            break;
        case Qt::Key_L:
            dx = 50;
            break;
        case Qt::Key_I:
            dy = -50;
            break;
        case Qt::Key_K:
            dy = 50;
            break;
        case Qt::Key_O:
            placeBomb();
            break;
        }
    }
    else if (playerNumber == 2) {
        switch (event->key()) {
        case Qt::Key_A:
            dx = -50;
            break;
        case Qt::Key_D:
            dx = 50;
            break;
        case Qt::Key_W:
            dy = -50;
            break;
        case Qt::Key_S:
            dy = 50;
            break;
        case Qt::Key_E:
            placeBomb();
            break;
        }
    }
    movePlayer(dx, dy);
}

void Player::placeBomb()
{
    Bomb* bomb = new Bomb(scene(), nullptr);
    bomb->setPos(x(), y());
    emit bombPlaced(bomb);
}

void Player::movePlayer(int dx, int dy)
{
    int newX = pos().x() + dx;
    int newY = pos().y() + dy;

    int minX = 0;
    int minY = 0;
    int maxX = 600;
    int maxY = 500;

    if (newX < minX || newX > maxX || newY < minY || newY > maxY) {
        return;
    }

    QList<QGraphicsItem*> itemsList = scene()->items();
    foreach (QGraphicsItem* item, itemsList) {
        Player* player = dynamic_cast<Player*>(item);
        if (player && player != this && player->pos() == QPointF(newX, newY)) {
            return;
        }

        IndestructibleWall* indestructibleWall = dynamic_cast<IndestructibleWall*>(item);
        if (indestructibleWall && indestructibleWall->pos() == QPointF(newX, newY)) {
            return;
        }

        DestructibleWall* destructibleWall = dynamic_cast<DestructibleWall*>(item);
        if (destructibleWall && destructibleWall->pos() == QPointF(newX, newY)) {
            return;
        }

        Bomb* bomb = dynamic_cast<Bomb*>(item);
        if (bomb && bomb->pos() == QPointF(newX, newY)) {
            return;
        }

        Explosion* explosion = dynamic_cast<Explosion*>(item);
        if (explosion && explosion->pos() == QPointF(newX, newY)) {
            return;
        }
    }

    setPos(newX, newY);
    emit playerMoved(newX, newY, playerNumber);
}

