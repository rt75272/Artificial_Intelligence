from flask import Blueprint, render_template, request, jsonify, redirect, url_for, session
from website_models import PRODUCT_MAP, PRODUCTS
from website_cart import cart_add, cart_remove, cart_clear, cart_items, cart_count
from website_config import settings
import smtplib
from email.message import EmailMessage

website_bp = Blueprint('website_bp', __name__, template_folder='templates')

@website_bp.route("/")
def home():
    return render_template("website/index.html", products=PRODUCTS, cart_count=cart_count())

@website_bp.route("/demo")
def demo():
    return render_template("website/demo.html")

@website_bp.route("/cart")
def view_cart():
    items = cart_items()
    enriched = []
    total = 0.0
    for item in items:
        product = PRODUCT_MAP.get(item["product_id"])
        if product:
            subtotal = product.price * item["qty"]
            total += subtotal
            enriched.append({
                "product": product,
                "qty": item["qty"],
                "subtotal": subtotal,
            })
    return render_template("website/cart.html", items=enriched, total=total, cart_count=cart_count())

@website_bp.route("/cart/add/<product_id>", methods=["POST"])
def add_to_cart(product_id):
    if product_id in PRODUCT_MAP:
        cart_add(product_id)
    return redirect(url_for("website_bp.home"))

@website_bp.route("/cart/remove/<product_id>", methods=["POST"])
def remove_from_cart(product_id):
    cart_remove(product_id)
    return redirect(url_for("website_bp.view_cart"))

@website_bp.route("/checkout")
def checkout_page():
    items = cart_items()
    enriched = []
    total = 0.0
    for item in items:
        product = PRODUCT_MAP.get(item["product_id"])
        if product:
            subtotal = product.price * item["qty"]
            total += subtotal
            enriched.append({"product": product, "qty": item["qty"], "subtotal": subtotal})

    if not enriched:
        return redirect(url_for("website_bp.view_cart"))

    sdk_ready = settings.paypal_client_id not in ("", "YOUR_PAYPAL_CLIENT_ID_HERE")

    return render_template("website/checkout.html",
                           items=enriched,
                           total=total,
                           cart_count=cart_count(),
                           paypal_client_id=settings.paypal_client_id,
                           paypal_env=settings.paypal_env,
                           sdk_ready=sdk_ready,
                           paypal_me_username=settings.paypal_me_username,
                           venmo_username=settings.venmo_username)

@website_bp.route("/checkout/complete", methods=["POST"])
def checkout_complete():
    body = request.get_json(silent=True) or {}
    order_id = body.get("orderID", "")
    cart_clear()
    return jsonify({"status": "ok", "order_id": order_id})

@website_bp.route("/order-success")
def order_success():
    return render_template("website/success.html", cart_count=0)

@website_bp.route("/demo/contact", methods=["POST"])
def demo_contact():
    body = request.get_json(silent=True) or {}
    fname = body.get("fname", "").strip()
    lname = body.get("lname", "").strip()
    email = body.get("email", "").strip()
    project = body.get("project", "").strip()

    if not all([fname, lname, email, project]):
        return jsonify({"error": "All fields are required."}), 422

    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Invalid email address."}), 422

    if not settings.smtp_user or not settings.smtp_password:
        return jsonify({"error": "Email sending is not configured yet. Set SMTP_USER and SMTP_PASSWORD in .env."}), 503

    msg = EmailMessage()
    msg["Subject"] = f"WebCraft Contact: {fname} {lname}"
    msg["From"] = settings.smtp_user
    msg["To"] = settings.contact_recipient
    msg["Reply-To"] = email
    msg.set_content(
        f"New contact form submission from the WebCraft demo site.\n\n"
        f"Name:    {fname} {lname}\n"
        f"Email:   {email}\n"
        f"Project:\n{project}\n"
    )

    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(settings.smtp_user, settings.smtp_password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        return jsonify({"error": "SMTP authentication failed. Check SMTP_USER and SMTP_PASSWORD in .env."}), 500
    except Exception as exc:
        return jsonify({"error": f"Could not send email: {exc}"}), 500

    return jsonify({"status": "ok"})
