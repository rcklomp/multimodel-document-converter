from mmrag_v2.processor import V2DocumentProcessor


class _BBox:
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b


class _Prov:
    def __init__(self, page_no, bbox):
        self.page_no = page_no
        self.bbox = bbox


class _Element:
    def __init__(self, page_no, bbox):
        self.prov = [_Prov(page_no=page_no, bbox=bbox)]


class _Page:
    def __init__(self, page_no, width):
        self.page_no = page_no
        self.width = width


class _Doc:
    def __init__(self, items, pages):
        self._items = items
        self.pages = pages

    def iterate_items(self):
        for it in self._items:
            yield it


def _processor_stub():
    # Avoid full constructor (Docling init not needed for pure ordering tests).
    return V2DocumentProcessor.__new__(V2DocumentProcessor)


def test_order_items_for_page_two_column_left_then_right_top_down():
    p = _processor_stub()

    # Intentionally mixed input order.
    records = [
        {"item": ("R2", None), "orig_index": 0, "bbox": (340, 100, 500, 150)},
        {"item": ("L2", None), "orig_index": 1, "bbox": (40, 120, 280, 170)},
        {"item": ("R1", None), "orig_index": 2, "bbox": (340, 20, 500, 70)},
        {"item": ("L3", None), "orig_index": 3, "bbox": (40, 210, 280, 260)},
        {"item": ("L1", None), "orig_index": 4, "bbox": (40, 10, 280, 60)},
        {"item": ("R3", None), "orig_index": 5, "bbox": (340, 220, 500, 270)},
    ]

    ordered = p._order_items_for_page(records, page_width=600)
    labels = [r["item"][0] for r in ordered]

    assert labels == ["L1", "L2", "L3", "R1", "R2", "R3"]


def test_order_items_for_page_single_column_sorts_by_y_then_x():
    p = _processor_stub()

    records = [
        {"item": ("C3", None), "orig_index": 0, "bbox": (80, 210, 180, 260)},
        {"item": ("C1", None), "orig_index": 1, "bbox": (120, 10, 220, 60)},
        {"item": ("C2_left", None), "orig_index": 2, "bbox": (60, 100, 160, 150)},
        {"item": ("C2_right", None), "orig_index": 3, "bbox": (160, 100, 260, 150)},
    ]

    ordered = p._order_items_for_page(records, page_width=320)
    labels = [r["item"][0] for r in ordered]

    assert labels == ["C1", "C2_left", "C2_right", "C3"]


def test_get_ordered_doc_items_orders_pages_and_respects_two_columns():
    p = _processor_stub()

    e_p2_r = _Element(page_no=2, bbox=_BBox(330, 10, 520, 60))
    e_p1_r = _Element(page_no=1, bbox=_BBox(330, 100, 520, 150))
    e_p1_l = _Element(page_no=1, bbox=_BBox(40, 20, 270, 70))
    e_p2_l = _Element(page_no=2, bbox=_BBox(40, 30, 270, 80))

    items = [
        (e_p2_r, None),
        (e_p1_r, None),
        (e_p1_l, None),
        (e_p2_l, None),
    ]
    pages = {1: _Page(1, width=600), 2: _Page(2, width=600)}
    doc = _Doc(items=items, pages=pages)

    ordered = p._get_ordered_doc_items(doc)

    # Page 1 first: left then right; then page 2: left then right.
    assert ordered == [
        (e_p1_l, None),
        (e_p1_r, None),
        (e_p2_l, None),
        (e_p2_r, None),
    ]
