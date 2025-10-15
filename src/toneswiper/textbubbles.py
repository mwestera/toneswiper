from PyQt6.QtCore import Qt, QPointF, QTimer
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsTextItem
from PyQt6.QtGui import QContextMenuEvent, QFont, QKeyEvent, QTextCursor, QColor, QBrush, QPen, QPolygonF
from .ui_helpers import measure

BASELINE_Y = 200
ROW_HEIGHT = 50
ARROW_WIDTH = 10
ARROW_HEIGHT = 3

class TextBubble(QGraphicsTextItem):
    """
    Defines an editable, movable text bubble.
    Holds .relative_x position (managed by .updateRelativeX and .moveToRelativeX), and
    .item_to_focus_next, a singleton list shared by all instances of TextBubble and their container,
    affecting tab/shift-tab behavior.
    """

    def __init__(self, text: str = "", item_to_focus_next: list = [None]):
        """
        Creates a TextBubble and customizes its looks a bit. The argument item_to_focus_next is intended
        to be a singleton list shared by all instances of TextBubble and their container, used higher up to
        customize tab/shift-tab behavior.
        """

        super().__init__(text)

        self.item_to_focus_next = item_to_focus_next
        self.relative_x = 0.0

        self.setFlags(
            QGraphicsTextItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsTextItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsTextItem.GraphicsItemFlag.ItemSendsGeometryChanges |
            QGraphicsTextItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)

        font = QFont("Arial", 18)
        self.setFont(font)
        self.setDefaultTextColor(Qt.GlobalColor.black)
        self.bg_color = QColor(210, 210, 210)
        self.margin = 4
        self.padding = 8
        self.setAcceptHoverEvents(True)

    def updateRelativeX(self):
        """
        Call whenever the item moves to keep relative_x in sync.
        """
        if self.scene():
            width = self.scene().sceneRect().width()
            if width > 0:
                self.relative_x = (self.pos().x() + self.boundingRect().width()/2) / width

    def moveToRelativeX(self, do_snap=True):
        """
        Place item at its stored relative_x and call snap().
        """
        if self.scene():
            width = self.scene().sceneRect().width()
            self.setX(self.relative_x * width - self.boundingRect().width()/2)
            self.updateRelativeX()
            if do_snap:
                self.snap()

    def boundingRect(self):
        """
        Overwrites super's method; original text rectangle + margin
        """
        rect = super().boundingRect()
        return rect.adjusted(-self.margin, - self.margin - ARROW_HEIGHT,
                             self.margin, self.margin)

    def paint(self, painter, option, widget=None):
        """
        Adds rounded rectangle with little pointer in the top center.
        """
        rect = self.boundingRect().adjusted(self.margin / 2, ARROW_HEIGHT + self.margin / 2, -self.margin / 2, -self.margin / 2)
        painter.setBrush(QBrush(self.bg_color))
        painter.setPen(QPen(Qt.GlobalColor.transparent))
        painter.drawRoundedRect(rect, 6, 6)

        left = QPointF(rect.x() + (rect.width() - rect.x()) / 2 - ARROW_WIDTH / 2, rect.y())
        right = QPointF(rect.x() + (rect.width() - rect.x()) / 2 + ARROW_WIDTH / 2, rect.y())
        top = QPointF(rect.x() + (rect.width() - rect.x()) / 2, rect.y() - ARROW_HEIGHT)
        painter.drawPolygon(QPolygonF([left, right, top]))

        super().paint(painter, option, widget)  # continues to draw the text

    def keyPressEvent(self, event: QKeyEvent):
        """
        Handles shift+left/right for movement, and enter/escape for deselection.
        """

        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            if event.key() == Qt.Key.Key_Left:
                self.moveBy(-0.01 * self.scene().sceneRect().width(), 0)
                self.updateRelativeX()
                self.register_shift_left_right()
                event.accept()
                return
            elif event.key() == Qt.Key.Key_Right:
                self.moveBy(0.01 * self.scene().sceneRect().width(), 0)
                self.updateRelativeX()
                self.register_shift_left_right()
                event.accept()
                return

        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape):
            self.clearFocus()
            event.accept()
            return

        super().keyPressEvent(event)
        self.moveToRelativeX(do_snap=False)  # Needed here because editing text changes where the center is.

    @measure
    def register_shift_left_right(self):  # for logging only
        pass

    @measure
    def contextMenuEvent(self, event: QContextMenuEvent):
        """
        Right click removes a TextBubble.
        """
        if self.scene():
            self.scene().removeItem(self)

    @measure
    def mouseMoveEvent(self, event):
        """
        LeftButton moves a TextBubble, updating first absolute position then relative_x.
        """
        if event.buttons() & Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            self.setPos(QPointF(pos.x() - self.boundingRect().width()/2, pos.y()))
            self.focusInEvent(None)
            self.updateRelativeX()
        else:
            super().mouseMoveEvent(event)

    @measure
    def focusOutEvent(self, event):
        """
        Focus out can be the result of many actions, some of which require deletion or splitting up
        of a TextBubble, and most of which require recalibrating (relative x, and snapping).
        """

        self.moveToRelativeX()

        cursor = self.textCursor()
        cursor.clearSelection()
        self.setTextCursor(cursor)

        self.setPlainText(self.toPlainText().strip())

        if not self.toPlainText():
            if self.scene():
                self.scene().removeItem(self)
            return

        if self.scene() and ' ' in self.toPlainText():
            self.split_on_spaces()
            return

        self.item_to_focus_next[0] = self
        self.snap()
        if event is not None:
            super().focusOutEvent(event)

    @measure
    def split_on_spaces(self):
        """
        Splits self on the spaces in the containing text, into several new TextBubble objects,
        adding them to the containing scene.
        """
        parts = [p for p in self.toPlainText().split() if p]
        scene = self.scene()
        x_cursor = self.pos().x()
        y_cursor = self.pos().y()
        scene.removeItem(self)
        for part in parts:
            new_item = TextBubble(part, self.item_to_focus_next)
            scene.addItem(new_item)
            new_item.setPos(x_cursor, y_cursor)
            new_item.snap()
            new_item.updateRelativeX()
            x_cursor += new_item.boundingRect().width() + self.padding + 2
            self.item_to_focus_next[0] = new_item

    def snap(self):
        """
        Positions a TextBubble to avoid overlap with other textbubbles, by
        stacking vertically if necessary (top-to-bottom).
        """
        scene = self.scene()
        if scene:

            self.setX(min(max(0, self.pos().x() + self.boundingRect().width()/2), scene.width()) - self.boundingRect().width()/2)
            self.updateRelativeX()

            x1 = self.pos().x()
            x2 = x1 + self.boundingRect().width()
            occupied = []
            for it in scene.items():
                if isinstance(it, TextBubble) and it is not self:
                    y = round(it.pos().y())
                    ox1 = it.pos().x() - self.padding
                    ox2 = ox1 + self.padding + it.boundingRect().width() + self.padding
                    occupied.append((y, ox1, ox2))

            new_y = BASELINE_Y + ROW_HEIGHT
            for offset in range(0, 4 * ROW_HEIGHT, ROW_HEIGHT):
                y_try = BASELINE_Y + offset
                overlap = False
                for oy, ox1, ox2 in occupied:
                    if oy == y_try and not (x2 < ox1 or x1 > ox2):
                        overlap = True
                        break
                if not overlap:
                    new_y = y_try
                    break

            self.setPos(QPointF(self.pos().x(), new_y))

    @measure
    def focusInEvent(self, event):
        """
        Focusing on a TextBubble selects all the containing text, for easy editing.
        """
        if event is not None:
            super().focusInEvent(event)
        QTimer.singleShot(0, self._select_all)  # otherwise it gets deselected again, timing issue?

    def _select_all(self):
        """
        Helper function because of above timing issue, selects all text in the TextBubble.
        """
        if not self.scene() or not self.hasFocus():
            return
        cursor = self.textCursor()
        cursor.select(QTextCursor.SelectionType.Document)
        self.setTextCursor(cursor)

    def __str__(self):
        return f"TextBubble({self.toPlainText()})"

class TextBubbleScene(QGraphicsScene):
    """
    QGraphicsScene contains TextBubble objects, handles creation of new TextBubble objects;
    Passes keypresses to an item if focused, otherwise ignores to be consumed higher up.
    Also a function for handling tabbing, as called from e.g. a TabInterceptor.
    Also a function for resizing, to be called from the containing View.
    """

    def __init__(self):
        super().__init__()
        self.item_to_focus_next = [None]

    def mouseDoubleClickEvent(self, event):
        """
        If on empty area, create a new text bubble.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            if not self.items(event.scenePos()):
                self.new_item_from_doubleclick(event.scenePos())
                return
        super().mousePressEvent(event)

    def new_item(self, pos: QPointF, text: str = "") -> None:
        """
        Create a new item at absolute position
        """
        item = TextBubble(text, self.item_to_focus_next)
        item.setPos(pos)
        self.addItem(item)
        item.setFocus()
        item.updateRelativeX()
        self.item_to_focus_next[0] = item

    @measure
    def new_item_from_doubleclick(self, pos: QPointF):
        self.new_item(pos)

    def new_item_relx(self, rel_x: float, text: str = "") -> None:
        """
        Create a new item at relative x position (e.g., for programmatic insertion).
        """
        item = TextBubble(text, self.item_to_focus_next)
        self.addItem(item)
        item.relative_x = rel_x
        item.moveToRelativeX()
        self.item_to_focus_next[0] = item
        if ' ' in item.toPlainText():
            item.split_on_spaces()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        Passes keypresses onto focused TextBubble item, otherwise ignores.
        """
        item = None
        if (focus_item := self.focusItem()) and isinstance(focus_item, TextBubble) and focus_item.hasFocus():
            # focusItem() can (apparently?) also be the last-focused one, hence the hasFocus() check.
            item = focus_item

        if item:
            super().keyPressEvent(event)
        else:
            event.ignore()

    @measure
    def handle_tabbing(self, backward: bool = False) -> None:
        """
        Meant to overwrite default tab/shift-tab behavior for setting focus to TextBubble objects,
        by being called from e.g. a TabInterceptor.
        Determines if a TextBubble currently has focus. If so, go to next/previous (by x-axis position).
        If not, makes use of the 'global' item_to_focus_next[0] item, so tab goes to the most
        recently edited and/or created TextBubble.
        If no such item exists, goes to first (tab) or last (shift-tab) item by x-axis position.
        """

        already_focused_textbox = None
        if (focus_item := self.focusItem()) and isinstance(focus_item, TextBubble) and focus_item.hasFocus():
            # focusItem() can (apparently?) also be the last-focused one, hence the hasFocus() check.
            already_focused_textbox = focus_item

        forward = not backward
        all_textboxes = sorted(
            [i for i in self.items() if isinstance(i, TextBubble)],
            key=lambda x: x.scenePos().x()
        )
        if not all_textboxes:
            return

        if already_focused_textbox:
            current_focus_idx = all_textboxes.index(already_focused_textbox)
            if forward:
                current_focus_idx = (current_focus_idx + 1) % len(all_textboxes)
            else:  # backward
                current_focus_idx = (current_focus_idx - 1) % len(all_textboxes)
        elif self.item_to_focus_next[0] in all_textboxes:
            current_focus_idx = all_textboxes.index(self.item_to_focus_next[0])
            if backward:
                current_focus_idx = (current_focus_idx - 1) % len(all_textboxes)
        else:
            if forward:
                current_focus_idx = 0
            else:  # backward
                current_focus_idx = len(all_textboxes) - 1

        all_textboxes[current_focus_idx].setFocus()

    def list_all_text_bubbles(self):
        return [item for item in self.items() if isinstance(item, TextBubble)]

    def __str__(self):
        return "TextBubbleScene"
