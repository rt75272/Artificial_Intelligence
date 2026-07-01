/* ── Typewriter effect ── */
const roles = [
  "UI / UX Designer",
  "Frontend Developer",
  "Full-Stack Engineer",
  "Creative Problem Solver",
];
let roleIdx = 0, charIdx = 0, deleting = false;
const tw = document.getElementById("typewriter");

function tick() {
  const current = roles[roleIdx];
  tw.textContent = deleting
    ? current.slice(0, charIdx--)
    : current.slice(0, charIdx++);

  let delay = deleting ? 50 : 80;

  if (!deleting && charIdx > current.length) {
    deleting = true;
    delay = 1800;
  } else if (deleting && charIdx < 0) {
    deleting = false;
    charIdx = 0;
    roleIdx = (roleIdx + 1) % roles.length;
    delay = 400;
  }
  setTimeout(tick, delay);
}
tick();

/* ── Animate stat counters when they scroll into view ── */
const statEls = document.querySelectorAll(".stat__num");
const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    const el = entry.target;
    const target = parseInt(el.dataset.target, 10);
    let current = 0;
    const step = Math.ceil(target / 40);
    const timer = setInterval(() => {
      current = Math.min(current + step, target);
      el.textContent = current;
      if (current >= target) clearInterval(timer);
    }, 35);
    counterObserver.unobserve(el);
  });
}, { threshold: 0.5 });
statEls.forEach(el => counterObserver.observe(el));

/* ── Animate skill bars when scrolled into view ── */
const barObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    entry.target.querySelectorAll(".bar__fill").forEach(bar => {
      bar.style.width = bar.style.getPropertyValue("--w");
    });
    barObserver.unobserve(entry.target);
  });
}, { threshold: 0.3 });
document.querySelectorAll(".skill-group").forEach(el => barObserver.observe(el));

/* ── Mobile burger menu ── */
document.getElementById("burgerBtn").addEventListener("click", () => {
  document.querySelector(".site-nav__links").classList.toggle("open");
});

/* ── Contact form — POST to backend, send real email ── */
document.getElementById("contactForm").addEventListener("submit", async (e) => {
  e.preventDefault();

  const form    = e.target;
  const btn     = form.querySelector("button[type=submit]");
  const success = document.getElementById("formSuccess");
  const errorEl = document.getElementById("formError");

  // Reset state
  success.style.display = "none";
  errorEl.style.display = "none";

  // Disable form while sending
  btn.disabled  = true;
  btn.textContent = "Sending…";

  const payload = {
    fname:   form.fname.value.trim(),
    lname:   form.lname.value.trim(),
    email:   form.email.value.trim(),
    project: form.project.value.trim(),
  };

  try {
    const res  = await fetch("/demo/contact", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });
    const data = await res.json();

    if (res.ok) {
      success.style.display = "block";
      form.querySelectorAll("input, textarea, button").forEach(el => el.disabled = true);
    } else {
      errorEl.textContent   = data.error || "Something went wrong. Please try again.";
      errorEl.style.display = "block";
      btn.disabled          = false;
      btn.textContent       = "Send Message ✉️";
    }
  } catch {
    errorEl.textContent   = "Network error. Please check your connection and try again.";
    errorEl.style.display = "block";
    btn.disabled          = false;
    btn.textContent       = "Send Message ✉️";
  }
});
