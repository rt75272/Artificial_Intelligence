from flask import session

def _get_raw() -> dict:
    raw = session.get("cart", {})
    return raw if isinstance(raw, dict) else {}

def cart_add(product_id: str) -> None:
    cart = _get_raw()
    cart[product_id] = cart.get(product_id, 0) + 1
    session["cart"] = cart
    session.modified = True

def cart_remove(product_id: str) -> None:
    cart = _get_raw()
    cart.pop(product_id, None)
    session["cart"] = cart
    session.modified = True

def cart_clear() -> None:
    session["cart"] = {}
    session.modified = True

def cart_items() -> list:
    cart = _get_raw()
    return [{"product_id": pid, "qty": qty} for pid, qty in cart.items()]

def cart_count() -> int:
    return sum(_get_raw().values())
