const form = document.getElementById("risk-form");
if (form) {
  form.addEventListener("submit", (event) => {
    const missing = [...form.querySelectorAll("[required]")].filter((el) => {
      if (el.type === "number") return el.value === "" || Number.isNaN(Number(el.value));
      return !el.value;
    });
    if (missing.length) {
      event.preventDefault();
      missing[0].focus();
      alert("Please complete every required field with a valid value.");
    }
  });
}
