def update_cart(cart, item, qty):
    cart[item] = qty
    return cart

cart = {
    "Laptop": 1,
    "Mouse": 2
}

print(update_cart(cart, "Keyboard", 1))
print(update_cart(cart, "Mouse", 3))