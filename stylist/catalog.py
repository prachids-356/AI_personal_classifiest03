"""Shared constants for catalog tagging."""
CATEGORIES = ["dress", "top", "t-shirt", "shirt", "kurta", "jeans", "trousers", "skirt",
              "jumpsuit", "jacket", "blazer", "cardigan", "sweater", "palazzo", "saree", "shorts", "leggings",
              "boxer briefs", "briefs", "underwear", "bra", "panties", "lingerie",
              "sneakers", "heels", "sandals", "boots", "flats", "watch", "handbag", "backpack",
              "jewelry", "sunglasses", "belt", "wallet"]
NON_APPAREL_CATEGORIES = {"sneakers", "heels", "sandals", "boots", "flats", "watch", "handbag",
                          "backpack", "jewelry", "sunglasses", "belt", "wallet", "boxer briefs",
                          "briefs", "underwear", "bra", "panties", "lingerie"}
CATEGORY_PROMPTS = {
    "shorts": "a product photo of casual outerwear shorts, not underwear",
    "boxer briefs": "a product photo of men's boxer briefs underwear, not outerwear shorts",
    "briefs": "a product photo of men's briefs underwear, not outerwear shorts",
    "underwear": "a product photo of underwear briefs, not outerwear shorts",
    "bra": "a product photo of a women's bra underwear",
    "panties": "a product photo of women's panties underwear",
    "lingerie": "a product photo of lingerie underwear",
    "blazer": "a product photo of a tailored blazer jacket",
    "cardigan": "a product photo of a knit cardigan sweater",
}
COLORS = ["black", "white", "blue", "red", "green", "yellow", "pink", "purple", "grey",
          "brown", "beige", "orange", "navy blue", "maroon", "floral"]
OCCASIONS = ["casual", "formal", "party", "ethnic", "sports", "streetwear", "wedding", "smart casual"]
