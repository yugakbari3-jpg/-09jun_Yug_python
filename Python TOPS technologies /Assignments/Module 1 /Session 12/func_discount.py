def get_discounted_price(price, discount_percent):
    final_price = price - (price * discount_percent / 100)
    return final_price

print(get_discounted_price(500, 10))
