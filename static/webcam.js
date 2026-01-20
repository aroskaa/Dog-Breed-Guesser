const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const faceForm = document.getElementById("face-form");

// AKTIFKAN KAMERA
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        alert("Camera access denied");
    });

// CAPTURE & KIRIM
function captureFace() {
    const username = document.getElementById("face-username").value;
    if (!username) {
        alert("Username wajib diisi");
        return;
    }

    document.getElementById("hidden-username").value = username;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const file = new File([blob], "face.jpg", { type: "image/jpeg" });

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        document.getElementById("face-image").files = dataTransfer.files;

        faceForm.submit();
    }, "image/jpeg");
}
