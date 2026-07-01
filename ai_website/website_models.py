from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Product:
    id: str
    name: str
    tagline: str
    price: float          # USD
    features: List[str]
    badge: str = ""       # e.g. "Popular"
    icon: str = "🌐"


# ──────────────────────────────────────────────
# Product catalogue
# ──────────────────────────────────────────────
PRODUCTS: List[Product] = [
    Product(
        id="starter",
        name="Starter Site",
        tagline="Get online in minutes",
        price=29.00,
        icon="🚀",
        features=[
            "1-page responsive design",
            "Contact form",
            "Mobile-friendly layout",
            "SSL certificate",
            "1 revision included",
        ],
    ),
    Product(
        id="portfolio",
        name="Portfolio Pro",
        tagline="Showcase your best work",
        price=49.00,
        icon="🎨",
        badge="Popular",
        features=[
            "Up to 5 pages",
            "Project gallery",
            "About & bio section",
            "Social media links",
            "3 revisions included",
            "SEO basics",
        ],
    ),
    Product(
        id="personal_brand",
        name="Personal Brand",
        tagline="Your complete online presence",
        price=99.00,
        icon="⭐",
        features=[
            "Up to 10 pages",
            "Blog / news section",
            "Newsletter signup",
            "Google Analytics",
            "Custom domain setup",
            "5 revisions included",
            "Priority support (30 days)",
        ],
    ),
    Product(
        id="enterprise",
        name="Full Presence",
        tagline="Everything, handled for you",
        price=199.00,
        icon="💎",
        features=[
            "Unlimited pages",
            "E-commerce ready",
            "Custom animations",
            "CMS integration",
            "Performance audit",
            "Unlimited revisions",
            "6-month support plan",
        ],
    ),
]

PRODUCT_MAP: dict[str, Product] = {p.id: p for p in PRODUCTS}
