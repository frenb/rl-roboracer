async function start() {
    const memory = new Memory(500);
    const model = new Model(
        [256],
        3 /* state size */,
        3 /* action size */,
        100 /* replay batch size */
        );
    
    const orchestrator = new Orchestrator(
        30 /* max steps per game */,
        model,
        memory,
        0.95 /* discount rate */,
        0.2, /* initial eps */
        );
    
    
    let game = 0;
    while (game < 500) {
        let totalReward = await orchestrator.run()
        log(`generation ${game}: ${totalReward}`);
        game++;
    }
    
    const orchestrator2 = new Orchestrator(
        30 /* max steps per game */,
        model,
        memory,
        0.95 /* discount rate */,
        0.0, /* initial eps */
        );
      
    let totalReward = await orchestrator2.run()
    log(`Final generation: ${totalReward}`);

        
}









/*
async function tf_hello_world() {
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Generate some synthetic data for training.
    const xs = tf.tensor2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 1]);
    const ys = tf.tensor2d([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23], [12, 1]);
    
    // Train the model using the data.
    model.fit(xs, ys, {epochs: 10}).then(() => {
      // Use the model to do inference on a data point the model hasn't seen before:
      model.predict(tf.tensor2d([100], [1, 1])).print();
      // Open the browser devtools to see the output
    });
}*/