async function start() {
    tf_hello_world();
}










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
}