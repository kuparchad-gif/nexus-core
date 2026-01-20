function updateViewers(state) {
    let video = state.video;
    let modelLeft = state.modelLeft;
    let modelRight = state.modelRight;

    let gradioVideo = document.getElementById("gradioVideo");
    let videoComponent = gradioVideo ? gradioVideo.querySelector("video") : null;

    if (videoComponent && document.getElementById("modelViewerLeft") && document.getElementById("modelViewerRight")) {

        videoComponent.setAttribute("muted", true);
        document.getElementById("modelViewerLeft").contentWindow.postMessage({ action: "loadModel", modelUrl: `https://gradio-model-viewer.s3.eu-west-1.amazonaws.com/models/${modelLeft}/${video}.glb` }, "*");
        document.getElementById("modelViewerRight").contentWindow.postMessage({ action: "loadModel", modelUrl: `https://gradio-model-viewer.s3.eu-west-1.amazonaws.com/models/${modelRight}/${video}.glb` }, "*");

        let loadedCount = 0;

        window.addEventListener("message", (event) => {
            if (event.data.status === "modelLoaded") {
                loadedCount++;
                if (loadedCount === 2) {
                    videoComponent.addEventListener("play", syncModelViewers);
                    videoComponent.addEventListener("pause", syncModelViewers);
                    videoComponent.addEventListener("timeupdate", syncModelViewers);

                    Array.from(document.getElementsByClassName('thumbnail-btn')).forEach(btn => btn.disabled = false);
                }
            }
            else if (event.data.status === "modelLoadError") {
                Array.from(document.getElementsByClassName('thumbnail-btn')).forEach(btn => btn.disabled = false);
            }
        });

    }
}

function syncModelViewers(event) {
    let videoComponent = event.target;
    let modelViewerLeft = document.getElementById("modelViewerLeft");
    let modelViewerRight = document.getElementById("modelViewerRight");

    if (!modelViewerLeft || !modelViewerRight) return;

    switch (event.type) {
        case "play":
            modelViewerLeft.contentWindow.postMessage({ action: "playAnimation" }, "*");
            modelViewerRight.contentWindow.postMessage({ action: "playAnimation" }, "*");
            break;
        case "pause":
            modelViewerLeft.contentWindow.postMessage({ action: "pauseAnimation" }, "*");
            modelViewerRight.contentWindow.postMessage({ action: "pauseAnimation" }, "*");
            break;
        case "timeupdate":
            let currentTime = videoComponent.currentTime;
            modelViewerLeft.contentWindow.postMessage({ action: "setAnimationTime", currentTime }, "*");
            modelViewerRight.contentWindow.postMessage({ action: "setAnimationTime", currentTime }, "*");    
            break;
    }
}
