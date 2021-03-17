async function start() {
    log("Starting...");
    let img = document.getElementById("camera/overhead");
    // Load the model.
    cocoSsd.load({base: 'mobilenet_v2'}).then(model => {
        // detect objects in the image.
        model.detect(img, 20 /* max boxes */ , 0.1 /* min score */).then(predictions => {
            log('Predictions: ' + JSON.stringify(predictions, null, 3));
            let ctx = img.getContext("2d");
            window.cameraAnnotations = [];
            predictions.forEach(p => {
                window.cameraAnnotations.push(p.bbox);
            })
        });
    });
}