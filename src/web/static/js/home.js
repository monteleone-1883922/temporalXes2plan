/**
 * Home page: XES file upload handler.
 */

document.getElementById("xes-upload").addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const errorEl = document.getElementById("import-error");
    const area = document.getElementById("import-area");
    errorEl.classList.add("js-hidden");
    area.classList.add("import-area-loading");

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/upload-log", { method: "POST", body: formData });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.error || resp.statusText);
        window.location.href = `/setup/${data.config_name}`;
    } catch (err) {
        errorEl.textContent = err.message;
        errorEl.classList.remove("js-hidden");
        area.classList.remove("import-area-loading");
    }

    e.target.value = "";
});
