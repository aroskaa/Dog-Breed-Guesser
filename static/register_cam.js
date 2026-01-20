const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

// AKTIFKAN KAMERA
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => {
        alert("Camera access denied");
    });

// CAPTURE WAJAH
function captureFace() {
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(blob => {
        const file = new File([blob], "face.jpg", { type: "image/jpeg" });

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);

        document.getElementById("face-image").files = dataTransfer.files;

        alert("Face captured successfully!");
    }, "image/jpeg");
}
