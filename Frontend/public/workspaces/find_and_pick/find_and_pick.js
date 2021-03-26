const TABLE_PERPENDICULAR = {x : -0.5, y : -0.5, z : 0.5, w : -0.5};

async function start() {
    const img = document.getElementById("OverheadCameraPlayer_video");

    // Load the model.
    const model = await cocoSsd.load({base: 'mobilenet_v2'});
    log("Model loaded...");
    
    var scene_data = await api.getSceneData();
    var initial_pose =  scene_data.data.effector_pose;
    
    while (true) {
        await api.sleep(2000);
        //clearAnnotations();
        
        let predictions = await model.detect(img, 20, 0.05);
        log(`predictions = ${JSON.stringify(predictions, null, 3)}`);
        //predictions.forEach(p => drawBox(p.bbox));
        
        let bananas = predictions.filter(p => p.class == 'banana');
        
        if (bananas.length == 0) {
            log("No bananas today. We sleep.")
            continue;
        }
        
        var bbox = bananas[0].bbox;
        
        log("Found a banana. This can not stand.")
        
        var corners = [
            { x: bbox[0], y: bbox[1] },
            { x: bbox[0] + bbox[2], y: bbox[1] +  bbox[3]},
            { x: bbox[0] + bbox[2], y: bbox[1] },
            { x: bbox[0], y: bbox[1] + bbox[3] },
        ];
        corners = corners.map(c => canvasToWorldPoint(img, c));
        corners.sort((a,b) => { return a.x**2 + a.y**2 - b.x**2 - b.y**2 });
        
    
        var goal_1 = {position: corners[0] , orientation: TABLE_PERPENDICULAR};
        var goal_2 = {position: corners[3] , orientation: TABLE_PERPENDICULAR};
       
        // Ready. 
        let plan = await api.getPlan(goal_1);
        await api.doTrajectory(plan.trajectory);
        
        // Push.
        plan = await api.getPlan(goal_2);
        await api.doTrajectory(plan.trajectory);
        
        // Return.
        plan = await api.getPlan(initial_pose);
        await api.doTrajectory(plan.trajectory);
    }
        
}

function canvasToWorldPoint(canvas, point) {
    const top_left = {x: 0.5, y: -0.5, z: 0.6};
    const bottom_right = {x: -0.5, y: 0.5, z: 0.6}
    const max_x = canvas.videoWidth;
    const max_y = canvas.videoHeight;
    
    let world_y = (point.x / max_x) * (bottom_right.x - top_left.x) + top_left.x;
    let world_x = (point.y / max_y) * (bottom_right.y - top_left.y) + top_left.y;
    
    return {x: world_x, y: world_y, z: 0.7}
}


function clearAnnotations() {
    let canvas = document.getElementById('OverheadCameraPlayer_annotations');
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height); 
}

function drawBox(bbox) {
    let canvas = document.getElementById('OverheadCameraPlayer_annotations');
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.strokeStyle = "red";
    ctx.rect(bbox[0], bbox[1], bbox[2], bbox[3]);
    ctx.stroke();
}