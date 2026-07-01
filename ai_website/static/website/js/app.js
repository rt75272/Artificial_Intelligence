// Small UX enhancements

// Animate "Add to Cart" buttons on submit.
// We listen on the *form* submit event (not the button click) so that the
// browser has already locked in the POST before we disable the button —
// disabling a submit button inside its own click handler cancels submission.
document.querySelectorAll("form").forEach(form => {
  form.addEventListener("submit", () => {
    const btn = form.querySelector("button[type=submit]");
    if (!btn) return;
    btn.textContent = "Added ✓";
    btn.disabled = true;
    // Re-enable in case the redirect is slow or the user navigates back
    setTimeout(() => {
      btn.textContent = "Add to Cart";
      btn.disabled = false;
    }, 2000);
  });
});
